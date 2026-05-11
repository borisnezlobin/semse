import Foundation

enum SearchError: Error, LocalizedError {
    case http(Int)
    case decoding
    case notRunning

    var errorDescription: String? {
        switch self {
        case .http(let code): return "API returned \(code)"
        case .decoding: return "Couldn't decode response"
        case .notRunning: return "Backend isn't running on localhost:8000. Start it with `uvicorn api.main:app`."
        }
    }
}

final class SearchClient {
    static let shared = SearchClient()

    let baseURL: URL
    private let session: URLSession

    init(baseURL: URL = URL(string: "http://127.0.0.1:8000")!) {
        self.baseURL = baseURL
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 15
        self.session = URLSession(configuration: config)
    }

    func search(query: String) async throws -> SearchResponse {
        let intent = await QueryRouter.shared.parse(query)
        var req = URLRequest(url: baseURL.appendingPathComponent("search"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONEncoder().encode(SearchBody(query: query, top_k: 8, intent: intent))

        let (data, resp): (Data, URLResponse)
        do {
            (data, resp) = try await session.data(for: req)
        } catch {
            throw SearchError.notRunning
        }
        guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
            throw SearchError.http((resp as? HTTPURLResponse)?.statusCode ?? -1)
        }
        do {
            return try JSONDecoder().decode(SearchResponse.self, from: data)
        } catch {
            throw SearchError.decoding
        }
    }

    func contactPhotoURL(key: String) -> URL {
        baseURL.appendingPathComponent("contact-photo").appendingPathComponent(key)
    }
}

private struct SearchBody: Encodable {
    let query: String
    let top_k: Int
    let intent: QueryIntentWire
}
