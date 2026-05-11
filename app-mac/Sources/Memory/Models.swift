import Foundation

struct ChunkMessage: Codable, Identifiable, Hashable {
    let sender: String
    let isFromMe: Bool
    let text: String
    let dateIso: String
    let contactKey: String?
    let known: Bool
    let isBestMatch: Bool

    var id: String { "\(dateIso)-\(sender)-\(text.prefix(20))" }

    enum CodingKeys: String, CodingKey {
        case sender
        case isFromMe = "is_from_me"
        case text
        case dateIso = "date_iso"
        case contactKey = "contact_key"
        case known
        case isBestMatch = "is_best_match"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sender = try c.decode(String.self, forKey: .sender)
        isFromMe = try c.decode(Bool.self, forKey: .isFromMe)
        text = try c.decode(String.self, forKey: .text)
        dateIso = try c.decode(String.self, forKey: .dateIso)
        contactKey = try c.decodeIfPresent(String.self, forKey: .contactKey)
        known = try c.decodeIfPresent(Bool.self, forKey: .known) ?? true
        isBestMatch = try c.decodeIfPresent(Bool.self, forKey: .isBestMatch) ?? false
    }
}

struct SourceResult: Codable, Identifiable, Hashable {
    let source: String
    let contactNames: [String]
    let dateStart: String
    let dateEnd: String
    let score: Double
    let messages: [ChunkMessage]
    let subject: String?
    let chatTitle: String?
    let snippet: String
    let imageURL: String?
    let imageCaption: String?
    let attachmentID: Int?

    var id: String { "\(source)-\(dateStart)-\(contactNames.joined())-\(attachmentID ?? messages.count)" }

    enum CodingKeys: String, CodingKey {
        case source
        case contactNames = "contact_names"
        case dateStart = "date_start"
        case dateEnd = "date_end"
        case score
        case messages
        case subject
        case chatTitle = "chat_title"
        case snippet
        case imageURL = "image_url"
        case imageCaption = "image_caption"
        case attachmentID = "attachment_id"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        source = try c.decode(String.self, forKey: .source)
        contactNames = try c.decode([String].self, forKey: .contactNames)
        dateStart = try c.decode(String.self, forKey: .dateStart)
        dateEnd = try c.decode(String.self, forKey: .dateEnd)
        score = try c.decode(Double.self, forKey: .score)
        messages = (try? c.decode([ChunkMessage].self, forKey: .messages)) ?? []
        subject = try c.decodeIfPresent(String.self, forKey: .subject)
        chatTitle = try c.decodeIfPresent(String.self, forKey: .chatTitle)
        snippet = (try? c.decode(String.self, forKey: .snippet)) ?? ""
        imageURL = try c.decodeIfPresent(String.self, forKey: .imageURL)
        imageCaption = try c.decodeIfPresent(String.self, forKey: .imageCaption)
        attachmentID = try c.decodeIfPresent(Int.self, forKey: .attachmentID)
    }
}

struct SearchResponse: Codable {
    let answer: String
    let sources: [SourceResult]
    let queryMs: Int

    enum CodingKeys: String, CodingKey {
        case answer
        case sources
        case queryMs = "query_ms"
    }
}
