import AppKit

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var initialPresentationAttempts = 0
    private var hasPresentedInitialWindow = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        presentMainWindowIfNeeded()
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            bringMainWindowToFront()
        }
        return true
    }

    private func presentMainWindowIfNeeded() {
        guard !hasPresentedInitialWindow else { return }

        initialPresentationAttempts += 1

        if bringMainWindowToFront() {
            hasPresentedInitialWindow = true
            return
        }

        guard initialPresentationAttempts < 20 else { return }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            self?.presentMainWindowIfNeeded()
        }
    }

    @discardableResult
    private func bringMainWindowToFront() -> Bool {
        NSApplication.shared.activate(ignoringOtherApps: true)

        guard let window = NSApplication.shared.windows.first else {
            return false
        }

        if window.isMiniaturized {
            window.deminiaturize(nil)
        }

        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
        return true
    }
}
