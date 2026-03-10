import Foundation

enum PrivilegedHelperConstants {
    static let helperLabel = "com.triglav.gui.helper"
    static let helperLaunchDaemonPlistName = "com.triglav.gui.helper.plist"
    static let helperExecutableName = "TriglavPrivilegedHelper"
    static let tunServiceLabel = "com.triglav.gui.tun"
    static let installedTriglavBinaryPath = "/Library/PrivilegedHelperTools/com.triglav.gui.triglav"
    static let installedLaunchDaemonPlistPath = "/Library/LaunchDaemons/\(tunServiceLabel).plist"
}

@objc protocol TriglavPrivilegedHelperXPCProtocol {
    func ping(_ reply: @escaping (String) -> Void)
    func installTunService(
        binarySourcePath: String,
        arguments: [String],
        logPath: String,
        withReply reply: @escaping (String?) -> Void
    )
    func uninstallTunService(withReply reply: @escaping (String?) -> Void)
}
