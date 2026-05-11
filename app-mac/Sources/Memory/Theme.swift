import SwiftUI

enum Theme {
    // System macOS blue, the iMessage send-bubble color.
    static let accent = Color(red: 10/255, green: 132/255, blue: 1.0)
    static let imessageBlue = Color(red: 35/255, green: 130/255, blue: 1.0)
    static let theirGray = Color(white: 0.22)
    static let theirGrayText = Color.white.opacity(0.95)

    // Source tint colors (used for the small platform pill).
    static let imessageGreen = Color(red: 48/255, green: 209/255, blue: 88/255)
    static let mailBlue = Color(red: 10/255, green: 132/255, blue: 1.0)

    static let panelStroke = Color.white.opacity(0.06)
    static let dividerLine = Color.white.opacity(0.06)
    static let secondaryText = Color.white.opacity(0.5)
    static let tertiaryText = Color.white.opacity(0.35)
    static let placeholder = Color.white.opacity(0.30)
}

extension Color {
    /// Initials/avatar background derived deterministically from a name.
    static func avatarBackground(for name: String) -> Color {
        let hues: [Double] = [0.02, 0.08, 0.16, 0.34, 0.55, 0.62, 0.78, 0.92]
        var hash = 5381
        for ch in name.unicodeScalars { hash = ((hash << 5) &+ hash) &+ Int(ch.value) }
        let h = hues[abs(hash) % hues.count]
        return Color(hue: h, saturation: 0.45, brightness: 0.6)
    }
}
