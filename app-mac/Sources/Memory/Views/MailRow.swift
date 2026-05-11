import SwiftUI

/// Renders a mail message as a sender + truncated body, not as a chat bubble.
/// Email bodies are too long for the iMessage bubble layout to look good.
struct MailRow: View {
    let message: ChunkMessage
    let expanded: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(message.sender)
                    .font(.system(size: 11.5, weight: .semibold))
                    .foregroundColor(.white.opacity(0.75))
                Spacer()
                if !message.dateIso.isEmpty {
                    Text(formatDate(message.dateIso))
                        .font(.system(size: 10.5))
                        .foregroundColor(Theme.tertiaryText)
                }
            }
            Text(message.text)
                .font(.system(size: 12.5))
                .foregroundColor(Theme.secondaryText)
                .lineLimit(expanded ? nil : 4)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.white.opacity(0.03))
        )
    }
}
