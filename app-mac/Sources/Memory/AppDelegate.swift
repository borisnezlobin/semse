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
    private let panelMinHeight: CGFloat = 56  // matches the input row exactly
    private let panelMaxHeight: CGFloat = 640
    // Clears the search state if the panel stays hidden this long. The timer
    // runs ONLY while the panel is hidden — a user who keeps Semse open and
    // reads results never gets reset.
    private let idleClearAfterSeconds: UInt64 = 60
    private var idleClearTask: Task<Void, Never>?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let panel = SpotlightPanel()
        panel.delegate = self
        let root = SearchView(
            onDismiss: { [weak self] in self?.hidePanel() },
            onContentHeightChange: { [weak self] h in self?.handleContentHeightChanged(h) }
        )
        let hosting = NSHostingView(rootView: root)
        hosting.sizingOptions = [.intrinsicContentSize]
        // Pin the hosting view directly into the blur container. We had an
        // NSScrollView wrapping it, but its document-view positioning fought
        // our manual Auto Layout constraints on the first layout pass, which
        // caused the content to render at ~50% offset within the panel frame.
        // The panel is capped at panelMaxHeight = 640; if results overflow,
        // SwiftUI clips. That's acceptable for now.
        hosting.translatesAutoresizingMaskIntoConstraints = false
        panel.blurContainer.addSubview(hosting)
        NSLayoutConstraint.activate([
            hosting.topAnchor.constraint(equalTo: panel.blurContainer.topAnchor),
            hosting.leadingAnchor.constraint(equalTo: panel.blurContainer.leadingAnchor),
            hosting.trailingAnchor.constraint(equalTo: panel.blurContainer.trailingAnchor),
        ])
        self.spotlightPanel = panel
        self.hosting = hosting
        // Force one layout pass so SwiftUI computes its intrinsic size before
        // the panel is shown; the first onPreferenceChange will then carry the
        // real content height instead of a transient zero/initial reading.
        hosting.layoutSubtreeIfNeeded()

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
        // Round to whole pixels — sub-pixel deltas were causing the resize
        // chain to occasionally no-op on legitimate height changes.
        if Int(panel.frame.height.rounded()) == Int(target.rounded()) { return }
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
        // Window shadow is cached against the previous alpha mask; force a
        // redraw so the shadow follows the new rounded bottom edge.
        panel.invalidateShadow()
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
        // Reopening within the idle window keeps the previous search.
        idleClearTask?.cancel()
        idleClearTask = nil
        // Do NOT reset the panel to compact height here. The panel's frame
        // already reflects the current SwiftUI content:
        //   - On first launch, init's contentRect was set to panelMinHeight.
        //   - If results are persisted from the previous open, panel is
        //     still at the matching height.
        //   - If the idle-clear notification fired while hidden, SearchView's
        //     state was wiped and SwiftUI re-emitted its smaller preference,
        //     shrinking the panel even before reopen.
        // Forcing compact here caused the "tiny sliver of results" bug after
        // close+reopen with a persisted query — SwiftUI's preferenceChange
        // never re-fires when its value hasn't changed.
        positionPanel(panel)
        NSApp.activate(ignoringOtherApps: true)
        panel.orderFrontRegardless()
        panel.makeKeyAndOrderFront(nil)
        panel.invalidateShadow()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
            NotificationCenter.default.post(name: .focusSearchField, object: nil)
        }
    }

    private func hidePanel() {
        spotlightPanel?.orderOut(nil)
        // Arm the idle clear timer. If the panel stays hidden for the full
        // window, we reset SearchView state; reopening before then cancels.
        idleClearTask?.cancel()
        let seconds = idleClearAfterSeconds
        idleClearTask = Task {
            try? await Task.sleep(nanoseconds: seconds * 1_000_000_000)
            if Task.isCancelled { return }
            await MainActor.run {
                NotificationCenter.default.post(name: .clearSearchState, object: nil)
            }
        }
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
