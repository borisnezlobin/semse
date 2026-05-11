import AppKit
import Carbon.HIToolbox
import SwiftUI

private func log(_ message: String) {
    FileHandle.standardError.write(Data("[Memory] \(message)\n".utf8))
}

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    private var spotlightPanel: SpotlightPanel?
    private var hosting: NSHostingView<SearchView>?
    private var hotKey: GlobalHotKey?
    private var statusItem: NSStatusItem?
    private let panelWidth: CGFloat = 720
    private let panelMinHeight: CGFloat = 64
    private let panelMaxHeight: CGFloat = 720

    func applicationDidFinishLaunching(_ notification: Notification) {
        let panel = SpotlightPanel()
        panel.delegate = self
        let root = SearchView(
            onDismiss: { [weak self] in self?.hidePanel() },
            onContentHeightChange: { [weak self] h in self?.handleContentHeightChanged(h) }
        )
        let hosting = NSHostingView(rootView: root)
        // Wrap the hosting view in an AppKit NSScrollView so long content can
        // scroll without SwiftUI's ScrollView machinery interfering with the
        // intrinsic-content-size loop.
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.hasHorizontalScroller = false
        scroll.scrollerStyle = .overlay
        scroll.drawsBackground = false
        scroll.contentView.drawsBackground = false
        scroll.translatesAutoresizingMaskIntoConstraints = false
        scroll.documentView = hosting
        // Pin hosting to the scroll view's edges so it auto-sizes width-wise.
        hosting.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            hosting.topAnchor.constraint(equalTo: scroll.contentView.topAnchor),
            hosting.leadingAnchor.constraint(equalTo: scroll.contentView.leadingAnchor),
            hosting.widthAnchor.constraint(equalTo: scroll.contentView.widthAnchor),
        ])
        panel.blurContainer.addSubview(scroll)
        NSLayoutConstraint.activate([
            scroll.topAnchor.constraint(equalTo: panel.blurContainer.topAnchor),
            scroll.leadingAnchor.constraint(equalTo: panel.blurContainer.leadingAnchor),
            scroll.trailingAnchor.constraint(equalTo: panel.blurContainer.trailingAnchor),
            scroll.bottomAnchor.constraint(equalTo: panel.blurContainer.bottomAnchor),
        ])
        self.spotlightPanel = panel
        self.hosting = hosting

        hotKey = GlobalHotKey(
            keyCode: UInt32(kVK_Space),
            modifiers: UInt32(controlKey | optionKey)
        ) { [weak self] in self?.togglePanel() }
        log("hotkey registered: Ctrl+Option+Space")

        installStatusItem()

        DispatchQueue.main.async { [weak self] in self?.showPanel() }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        showPanel()
        return true
    }

    private func handleContentHeightChanged(_ height: CGFloat) {
        guard let panel = spotlightPanel else { return }
        guard height > 1 else { return }  // ignore the no-content reading
        let target = max(panelMinHeight, min(panelMaxHeight, height))
        if abs(panel.frame.height - target) < 0.5 { return }
        // Keep the top edge fixed: in AppKit the y origin is the BOTTOM of the
        // window, so growing height means lowering origin.y so the top stays put.
        let oldFrame = panel.frame
        let newFrame = NSRect(
            x: oldFrame.origin.x,
            y: oldFrame.origin.y + (oldFrame.height - target),
            width: panelWidth,
            height: target
        )
        panel.setFrame(newFrame, display: true, animate: false)
    }

    // MARK: - Status item

    private func installStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = item.button {
            let image = NSImage(systemSymbolName: "magnifyingglass.circle", accessibilityDescription: "Memory")
            image?.isTemplate = true
            button.image = image
            button.target = self
            button.action = #selector(statusItemClicked)
        }
        let menu = NSMenu()
        menu.addItem(withTitle: "Search…  (⌃⌥Space)", action: #selector(togglePanelAction), keyEquivalent: "")
            .target = self
        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit", action: #selector(quitAction), keyEquivalent: "q")
            .target = self
        item.menu = menu
        self.statusItem = item
    }

    @objc private func statusItemClicked() { togglePanel() }
    @objc private func togglePanelAction() { togglePanel() }
    @objc private func quitAction() { NSApp.terminate(nil) }

    // MARK: - Panel control

    private func togglePanel() {
        guard let panel = spotlightPanel else { return }
        if panel.isVisible { hidePanel() } else { showPanel() }
    }

    private func showPanel() {
        guard let panel = spotlightPanel else { return }
        positionPanel(panel)
        NSApp.activate(ignoringOtherApps: true)
        panel.orderFrontRegardless()
        panel.makeKeyAndOrderFront(nil)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
            NotificationCenter.default.post(name: .focusSearchField, object: nil)
        }
    }

    private func hidePanel() {
        spotlightPanel?.orderOut(nil)
    }

    private func positionPanel(_ panel: NSPanel) {
        guard let screen = NSScreen.main else { return }
        let panelSize = panel.frame.size
        let v = screen.visibleFrame
        let x = v.midX - panelSize.width / 2
        let y = v.maxY - panelSize.height - v.height * 0.22
        panel.setFrameOrigin(NSPoint(x: x, y: y))
    }

    func windowDidResignKey(_ notification: Notification) {
        hidePanel()
    }
}

extension Notification.Name {
    // Defined on Notification.Name in SearchView.swift:
    // static let focusSearchField
}
