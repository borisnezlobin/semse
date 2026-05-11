import SwiftUI

struct Avatar: View {
    let name: String
    let contactKey: String?
    var size: CGFloat = 30

    @State private var image: NSImage?

    var body: some View {
        ZStack {
            if let image = image {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFill()
            } else {
                Color.avatarBackground(for: name)
                Text(initials(from: name))
                    .font(.system(size: size * 0.40, weight: .semibold, design: .rounded))
                    .foregroundColor(.white.opacity(0.92))
            }
        }
        .frame(width: size, height: size)
        .clipShape(Circle())
        .task(id: contactKey) { await loadPhoto() }
    }

    private func loadPhoto() async {
        guard let key = contactKey, !key.isEmpty else {
            image = nil
            return
        }
        let url = SearchClient.shared.contactPhotoURL(key: key)
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            if let img = NSImage(data: data) {
                await MainActor.run { self.image = img }
            }
        } catch {
            // Silent; we just fall back to initials.
        }
    }

    private func initials(from name: String) -> String {
        let parts = name.split(separator: " ").prefix(2)
        let chars = parts.compactMap { $0.first }
        if chars.isEmpty { return "?" }
        return String(chars).uppercased()
    }
}
