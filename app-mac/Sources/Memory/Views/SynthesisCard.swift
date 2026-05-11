import SwiftUI

struct SynthesisCard: View {
    let answer: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "sparkle")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(Theme.accent)
                Text("Answer")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(Theme.tertiaryText)
            }
            Text(.init(answer))
                .font(.system(size: 13.5))
                .foregroundColor(.white.opacity(0.88))
                .lineSpacing(4)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color.white.opacity(0.04))
        )
    }
}
