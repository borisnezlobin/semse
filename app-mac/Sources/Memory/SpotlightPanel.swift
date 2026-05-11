import AppKit

/// Frameless floating panel — uses NSVisualEffectView as the content view.
/// Note on macOS 26 (Tahoe): combining `.canJoinAllSpaces` with `.moveToActiveSpace`
/// in collectionBehavior hangs during init. Use them mutually-exclusively.
final class SpotlightPanel: NSPanel {
    let blurContainer: NSVisualEffectView

    init() {
        let contentRect = NSRect(x: 0, y: 0, width: 720, height: 56)

        let blur = NSVisualEffectView(frame: contentRect)
        blur.material = .hudWindow
        blur.blendingMode = .behindWindow
        blur.state = .active
        blur.wantsLayer = true
        blur.layer?.cornerRadius = 18
        blur.layer?.cornerCurve = .continuous
        blur.layer?.masksToBounds = true
        blur.autoresizingMask = [.width, .height]
        self.blurContainer = blur

        super.init(
            contentRect: contentRect,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        self.isOpaque = false
        self.backgroundColor = .clear
        self.hasShadow = true
        self.level = .floating
        self.isMovableByWindowBackground = false
        self.isMovable = false
        self.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        self.hidesOnDeactivate = false
        self.isReleasedWhenClosed = false
        self.animationBehavior = .utilityWindow
        self.contentView = blur
    }

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
    override var acceptsFirstResponder: Bool { true }
}
