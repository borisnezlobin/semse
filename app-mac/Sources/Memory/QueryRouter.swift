import Foundation
import FoundationModels

/// Structured intent extracted from the user's raw query.
/// Sent alongside the original query to the backend so retrieval can apply
/// hard filters (source / contact / attachment) and embed only the semantic
/// core of the query rather than the full natural-language string.
@Generable
struct QueryIntent: Codable, Sendable {
    @Guide(description: "The semantic core of the query, stripped of filter words like 'picture of', 'email about', 'message from', or contact names. This is what gets embedded for similarity search. If the query has no filter words, repeat the query unchanged.")
    var topic: String

    @Guide(description: "Source types the user explicitly named. Use 'imessage' for queries about text messages or iMessage, 'mail' for emails, 'image' for photos/pictures. Leave empty if the user didn't specify a source.")
    var sources: [SourceType]

    @Guide(description: "Contact names mentioned in the query (e.g. 'Jerry Yan', 'sarah'). Extract the name as written. Leave empty if no contact is named.")
    var contacts: [String]

    @Guide(description: "True only when the query explicitly asks for an image, photo, picture, screenshot, or attachment. False for general searches.")
    var mustHaveAttachment: Bool

    @Generable
    enum SourceType: String, Codable, Sendable {
        case imessage
        case mail
        case image
    }
}

/// JSON-encodable wire format the server understands.
struct QueryIntentWire: Encodable {
    let topic: String
    let sources: [String]
    let contacts: [String]
    let must_have_attachment: Bool

    init(_ intent: QueryIntent) {
        self.topic = intent.topic.trimmingCharacters(in: .whitespacesAndNewlines)
        self.sources = intent.sources.map(\.rawValue)
        self.contacts = intent.contacts
        self.must_have_attachment = intent.mustHaveAttachment
    }

    init(topic: String, sources: [String], contacts: [String], must_have_attachment: Bool) {
        self.topic = topic
        self.sources = sources
        self.contacts = contacts
        self.must_have_attachment = must_have_attachment
    }

    /// Passthrough used when the on-device model is unavailable or fails.
    /// Embedding the raw query, no filters applied — preserves current behavior.
    static func passthrough(query: String) -> QueryIntentWire {
        QueryIntentWire(
            topic: query.trimmingCharacters(in: .whitespacesAndNewlines),
            sources: [],
            contacts: [],
            must_have_attachment: false
        )
    }
}

private func logRouter(_ message: String) {
    FileHandle.standardError.write(Data("[QueryRouter] \(message)\n".utf8))
}

private let routerInstructions = """
Parse personal-search queries into a typed intent. The user searches their own messages, emails, and photos.

Rules:
- topic: the semantic core to embed. Strip filler ("picture of", "email about", "from X") and contact names. Keep temporal words ("recently", "yesterday"). Never return ONLY a temporal word; if that's all that's left, return the original query.
- sources: ["mail"] for email/mail/gmail; ["imessage"] for text/texts/message/messages/iMessage; ["image"] for photo/picture/pic/image/screenshot/gif. Empty if not specified.
- contacts: PEOPLE only. Never companies/brands/products/places (lockheed, amazon, google, openai, claude, chatgpt are NOT contacts). Lowercase first names fine.
- mustHaveAttachment: true iff query contains photo/picture/pic/image/screenshot/gif.

Examples:
"picture of white car" → topic="white car", sources=[image], mustHaveAttachment=true
"email about calculus" → topic="calculus", sources=[mail]
"does alarm prefer claude or chatgpt?" → topic="prefer claude or chatgpt", contacts=["alarm"]
"what did alex say about the trip" → topic="the trip", contacts=["alex"]
"who did I meet recently?" → topic="who did I meet recently"
"offer from lockheed" → topic="offer from lockheed", sources=[mail], contacts=[] (lockheed is a company)
"""

/// Parses raw queries into QueryIntent using Apple's on-device Foundation Model.
/// Caches recent parses, falls back to passthrough on any failure.
actor QueryRouter {
    static let shared = QueryRouter()

    private let session: LanguageModelSession?
    private var cache: [String: QueryIntentWire] = [:]
    private let cacheLimit = 64
    // The "tail" of a chain of pending parses. Each new parse captures the
    // current tail synchronously (inside this actor, no await between read
    // and update), creates a task that waits on the predecessor, and writes
    // itself as the new tail. This guarantees session.respond runs strictly
    // serially even if many callers arrive while one is mid-flight.
    private var tail: Task<Void, Never> = Task {}

    init() {
        if SystemLanguageModel.default.isAvailable {
            self.session = LanguageModelSession(instructions: routerInstructions)
            self.session?.prewarm()
        } else {
            self.session = nil
            logRouter("on-device model unavailable; using passthrough")
        }
    }

    func parse(_ query: String) async -> QueryIntentWire {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return QueryIntentWire.passthrough(query: trimmed)
        }
        if let cached = cache[trimmed] {
            return cached
        }
        guard let session else {
            return QueryIntentWire.passthrough(query: trimmed)
        }
        // Capture this call's predecessor and atomically chain. Multiple
        // callers reading tail in sequence each get a distinct predecessor
        // because the read+update happens without an intervening await.
        let predecessor = tail
        let resultTask = Task<QueryIntentWire, Never> { [weak self] in
            _ = await predecessor.value  // wait for full settle of prior parse
            // Belt-and-suspenders: poll isResponding before the next call.
            await Self.waitUntilReady(session)
            do {
                let response = try await session.respond(
                    to: "Parse this query: \(trimmed)",
                    generating: QueryIntent.self
                )
                let wire = QueryIntentWire(response.content)
                let safe = await self?.sanitize(wire, originalQuery: trimmed)
                    ?? QueryIntentWire.passthrough(query: trimmed)
                await self?.store(query: trimmed, intent: safe)
                logRouter(
                    "parsed: \"\(trimmed)\" → topic=\"\(safe.topic)\" " +
                    "sources=\(safe.sources) contacts=\(safe.contacts) " +
                    "attachment=\(safe.must_have_attachment)"
                )
                return safe
            } catch {
                logRouter("parse failed: \(error.localizedDescription); falling back to passthrough")
                return QueryIntentWire.passthrough(query: trimmed)
            }
        }
        tail = Task { _ = await resultTask.value }
        return await resultTask.value
    }

    /// Spin briefly until the language model session reports it's no longer
    /// responding to a previous prompt. Bounded to ~1.5s of polling so a
    /// stuck session can't lock the search bar.
    private static func waitUntilReady(_ session: LanguageModelSession) async {
        for _ in 0..<75 {
            if !session.isResponding { return }
            try? await Task.sleep(nanoseconds: 20_000_000) // 20ms
        }
    }

    /// Guard against degenerate parses: empty topic, or a topic the model
    /// reduced to only a temporal qualifier ("recently", "yesterday", etc).
    /// Temporal phrases are handled by the server's temporal.py — if they're
    /// the only thing left in the topic, the embedding has no signal.
    private func sanitize(_ wire: QueryIntentWire, originalQuery: String) -> QueryIntentWire {
        var topic = wire.topic.isEmpty ? originalQuery : wire.topic
        let normalized = topic.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if Self.bareTemporalTokens.contains(normalized) {
            topic = originalQuery
        }
        // Strip any contact entries that are actually temporal words the
        // model mistakenly classified as names.
        let contacts = wire.contacts.filter { !Self.bareTemporalTokens.contains($0.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)) }
        return QueryIntentWire(
            topic: topic,
            sources: wire.sources,
            contacts: contacts,
            must_have_attachment: wire.must_have_attachment
        )
    }

    private static let bareTemporalTokens: Set<String> = [
        "recently", "today", "yesterday", "tomorrow", "tonight",
        "week", "month", "year", "this week", "last week", "next week",
        "this month", "last month", "this year", "last year",
        "now", "ago", "soon",
    ]

    private func store(query: String, intent: QueryIntentWire) {
        if cache.count >= cacheLimit {
            cache.removeAll(keepingCapacity: true)
        }
        cache[query] = intent
    }
}

