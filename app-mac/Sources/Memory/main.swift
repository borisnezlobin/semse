import AppKit

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
// Accessory policy: no Dock icon. Menu-bar status item launches the panel.
app.setActivationPolicy(.accessory)
app.run()
