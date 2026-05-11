import AppKit
import SwiftUI

/// Renders an image search result: thumbnail, sender row, optional caption.
/// Click opens the underlying file with the system's default viewer.
struct ImageResultCard: View {
    let result: SourceResult
    let highlighted: Bool

    @State private var image: NSImage?
    @State private var loadFailed = false

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            thumbnail
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Avatar(name: result.contactNames.first ?? "Unknown", contactKey: nil, size: 22)
                    Text(headerLine)
                        .font(.system(size: 12.5, weight: .medium))
                        .foregroundColor(.white.opacity(0.9))
                        .lineLimit(1)
                    Spacer()
                    Text(formatDate(result.dateStart))
                        .font(.system(size: 10.5))
                        .foregroundColor(Theme.tertiaryText)
                }
                if let caption = result.imageCaption, !caption.isEmpty {
                    Text(caption)
                        .font(.system(size: 12))
                        .foregroundColor(Theme.secondaryText)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Text("Image · CLIP match")
                    .font(.system(size: 10.5))
                    .foregroundColor(Theme.tertiaryText.opacity(0.7))
            }
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(highlighted ? Color.white.opacity(0.06) : Color.white.opacity(0.02))
        )
        .task(id: result.imageURL) { await loadImage() }
        .onTapGesture(count: 2) { openInViewer() }
    }

    @ViewBuilder
    private var thumbnail: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.white.opacity(0.04))
                .frame(width: 88, height: 88)
            if let image = image {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 88, height: 88)
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            } else if loadFailed {
                Image(systemName: "photo")
                    .font(.system(size: 24))
                    .foregroundColor(Theme.tertiaryText)
            } else {
                ProgressView()
                    .controlSize(.small)
                    .progressViewStyle(.circular)
            }
        }
    }

    private var headerLine: String {
        if let chat = result.chatTitle, !chat.isEmpty {
            return "\(result.contactNames.first ?? "Unknown") · \(chat)"
        }
        return result.contactNames.first ?? "Unknown"
    }

    private func loadImage() async {
        guard let urlPath = result.imageURL else { return }
        let url = SearchClient.shared.baseURL.appendingPathComponent(
            urlPath.hasPrefix("/") ? String(urlPath.dropFirst()) : urlPath
        )
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            if let img = NSImage(data: data) {
                await MainActor.run { self.image = img }
            } else {
                await MainActor.run { self.loadFailed = true }
            }
        } catch {
            await MainActor.run { self.loadFailed = true }
        }
    }

    private func openInViewer() {
        guard let urlPath = result.imageURL else { return }
        let url = SearchClient.shared.baseURL.appendingPathComponent(
            urlPath.hasPrefix("/") ? String(urlPath.dropFirst()) : urlPath
        )
        NSWorkspace.shared.open(url)
    }
}
