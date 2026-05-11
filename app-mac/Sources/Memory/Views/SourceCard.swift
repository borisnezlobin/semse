import SwiftUI

struct SourceCard: View {
    let result: SourceResult
    let highlighted: Bool

    @State private var expanded = false

    private let collapsedShown = 3

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            header
            messagesView
            if hasMore && !expanded {
                expandButton
            } else if expanded && hasMore {
                collapseButton
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(highlighted ? Color.white.opacity(0.06) : Color.white.opacity(0.02))
        )
    }

    private var header: some View {
        HStack(spacing: 10) {
            Avatar(name: primaryName, contactKey: primaryKey, size: 28)
            VStack(alignment: .leading, spacing: 1) {
                Text(displayTitle)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.white.opacity(0.92))
                    .lineLimit(1)
                Text(metaLine)
                    .font(.system(size: 11))
                    .foregroundColor(Theme.tertiaryText)
                    .lineLimit(1)
            }
            Spacer()
        }
    }

    private var messagesView: some View {
        let (visible, hiddenAbove, hiddenBelow) = visibleWindow()
        return VStack(spacing: 4) {
            if hiddenAbove > 0 && !expanded {
                contextRow(text: "\(hiddenAbove) earlier message\(hiddenAbove == 1 ? "" : "s")")
            }
            ForEach(Array(visible.enumerated()), id: \.offset) { idx, msg in
                if result.source == "mail" {
                    MailRow(message: msg, expanded: expanded)
                } else {
                    MessageBubble(
                        message: msg,
                        showSender: idx == 0 || visible[idx - 1].sender != msg.sender
                    )
                }
            }
            if hiddenBelow > 0 && !expanded {
                contextRow(text: "\(hiddenBelow) later message\(hiddenBelow == 1 ? "" : "s")")
            }
        }
    }

    /// Center the visible window on the best-matching message (or the chunk
    /// tail if none is flagged). Returns (visibleSlice, hiddenBefore, hiddenAfter).
    private func visibleWindow() -> ([ChunkMessage], Int, Int) {
        if expanded { return (result.messages, 0, 0) }
        let count = result.messages.count
        if count <= collapsedShown { return (result.messages, 0, 0) }
        let anchor = result.messages.firstIndex(where: { $0.isBestMatch }) ?? (count - 1)
        // Window of `collapsedShown` messages centered on the anchor, clamped.
        let half = collapsedShown / 2
        var start = max(0, anchor - half)
        var end = min(count, start + collapsedShown)
        if end - start < collapsedShown {
            start = max(0, end - collapsedShown)
        }
        return (Array(result.messages[start..<end]), start, count - end)
    }

    private var expandButton: some View {
        Button(action: { withAnimation(.easeInOut(duration: 0.18)) { expanded = true } }) {
            HStack(spacing: 6) {
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 10, weight: .semibold))
                Text("Show full conversation (\(result.messages.count) messages)")
                    .font(.system(size: 11.5, weight: .medium))
            }
            .foregroundColor(Theme.accent)
            .padding(.vertical, 5)
            .padding(.horizontal, 10)
            .frame(maxWidth: .infinity)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color.white.opacity(0.04))
            )
        }
        .buttonStyle(.plain)
    }

    private var collapseButton: some View {
        Button(action: { withAnimation(.easeInOut(duration: 0.18)) { expanded = false } }) {
            HStack(spacing: 6) {
                Image(systemName: "chevron.up")
                    .font(.system(size: 10, weight: .semibold))
                Text("Collapse")
                    .font(.system(size: 11.5, weight: .medium))
            }
            .foregroundColor(Theme.tertiaryText)
            .padding(.vertical, 5)
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.plain)
    }

    private func contextRow(text: String) -> some View {
        HStack {
            line
            Text(text)
                .font(.system(size: 10.5, weight: .medium))
                .foregroundColor(Theme.tertiaryText)
            line
        }
        .padding(.vertical, 2)
    }

    private var line: some View {
        Rectangle().fill(Theme.dividerLine).frame(height: 1)
    }

    // MARK: - Derived

    private var hasMore: Bool { result.messages.count > collapsedShown }

    private var displayTitle: String {
        if let chat = result.chatTitle, !chat.isEmpty { return chat }
        if let subject = result.subject, !subject.isEmpty { return subject }
        return result.contactNames.prefix(2).joined(separator: ", ")
    }

    private var primaryName: String {
        result.contactNames.first ?? "Unknown"
    }

    private var primaryKey: String? {
        result.messages.first(where: { !$0.isFromMe })?.contactKey
    }

    private var metaLine: String {
        let label = SourceLabels.label(for: result.source)
        let date = formatDate(result.dateStart)
        return "\(label) · \(date)"
    }
}

enum SourceLabels {
    static func label(for source: String) -> String {
        switch source {
        case "imessage": return "iMessage"
        case "mail": return "Mail"
        case "hyperspell": return "Hyperspell"
        default: return source.capitalized
        }
    }
}

func formatDate(_ iso: String) -> String {
    if iso.isEmpty { return "" }
    let f = ISO8601DateFormatter()
    f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    var date = f.date(from: iso)
    if date == nil {
        f.formatOptions = [.withInternetDateTime]
        date = f.date(from: iso)
    }
    if date == nil {
        let alt = DateFormatter()
        alt.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        alt.locale = Locale(identifier: "en_US_POSIX")
        date = alt.date(from: iso)
    }
    guard let d = date else { return String(iso.prefix(10)) }
    let out = DateFormatter()
    out.dateStyle = .medium
    return out.string(from: d)
}
