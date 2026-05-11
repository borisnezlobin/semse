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
You parse personal-search queries into a typed intent. The user is searching their own messages, emails, and photos.

Rules:
- topic: the semantic content to embed. Strip filter words ("picture of", "email about", "messages from", "photo I sent to") and contact names. If the query is already just a topic ("white car", "thanksgiving plans"), keep it as-is. Never leave topic empty — if the query is only filters, return the most descriptive remaining noun phrase, or the raw query as a last resort.
- sources: only fill when explicit. "email" / "mail" → mail. "iMessage" / "text" / "messages" → imessage. "photo" / "picture" / "image" / "screenshot" → image. Don't infer source from topic words alone.
- contacts: extract proper names the user names. Lowercase first names are fine ("sarah"). Don't fabricate.
- mustHaveAttachment: true only when the query explicitly asks for visual content (photo, picture, image, screenshot).

Examples:
"white car" → topic: "white car", sources: [], contacts: [], mustHaveAttachment: false
"picture of white car" → topic: "white car", sources: [image], contacts: [], mustHaveAttachment: true
"photo of white car I sent to jerry yan" → topic: "white car", sources: [image], contacts: ["jerry yan"], mustHaveAttachment: true
"email about multivariable calculus" → topic: "multivariable calculus", sources: [mail], contacts: [], mustHaveAttachment: false
"email where I was introduced to someone" → topic: "introduced to someone", sources: [mail], contacts: [], mustHaveAttachment: false
"messages from sarah about thanksgiving" → topic: "thanksgiving", sources: [imessage], contacts: ["sarah"], mustHaveAttachment: false
"""

/// Parses raw queries into QueryIntent using Apple's on-device Foundation Model.
/// Caches recent parses, falls back to passthrough on any failure.
actor QueryRouter {
    static let shared = QueryRouter()

    private let session: LanguageModelSession?
    private var cache: [String: QueryIntentWire] = [:]
    private let cacheLimit = 64

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
        do {
            let response = try await session.respond(
                to: "Parse this query: \(trimmed)",
                generating: QueryIntent.self
            )
            let wire = QueryIntentWire(response.content)
            let safe = sanitize(wire, originalQuery: trimmed)
            store(query: trimmed, intent: safe)
            return safe
        } catch {
            logRouter("parse failed: \(error.localizedDescription); falling back to passthrough")
            return QueryIntentWire.passthrough(query: trimmed)
        }
    }

    /// Guard against degenerate parses: empty topic, or a topic the model
    /// returned literally as the filter words ("email", "picture").
    private func sanitize(_ wire: QueryIntentWire, originalQuery: String) -> QueryIntentWire {
        let topic = wire.topic.isEmpty ? originalQuery : wire.topic
        return QueryIntentWire(
            topic: topic,
            sources: wire.sources,
            contacts: wire.contacts,
            must_have_attachment: wire.must_have_attachment
        )
    }

    private func store(query: String, intent: QueryIntentWire) {
        if cache.count >= cacheLimit {
            cache.removeAll(keepingCapacity: true)
        }
        cache[query] = intent
    }
}

