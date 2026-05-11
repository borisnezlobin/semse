import SwiftUI

struct MessageBubble: View {
    let message: ChunkMessage
    let showSender: Bool   // first message in a run from this sender

    var body: some View {
        HStack(alignment: .bottom, spacing: 6) {
            if message.isFromMe {
                Spacer(minLength: 40)
                bubble
            } else {
                bubble
                Spacer(minLength: 40)
            }
        }
    }

    private var bubble: some View {
        VStack(alignment: message.isFromMe ? .trailing : .leading, spacing: 2) {
            if showSender && !message.isFromMe {
                Text(message.sender)
                    .font(.system(size: 10.5, weight: .medium))
                    .foregroundColor(Theme.tertiaryText)
                    .padding(.leading, 12)
            }
            Text(message.text)
                .font(.system(size: 13.5))
                .foregroundColor(message.isFromMe ? .white : Theme.theirGrayText)
                .padding(.horizontal, 11)
                .padding(.vertical, 7)
                .background(
                    bubbleShape
                        .fill(message.isFromMe ? Theme.imessageBlue : Theme.theirGray)
                )
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var bubbleShape: some Shape {
        RoundedRectangle(cornerRadius: 14, style: .continuous)
    }
}
