import Foundation

enum LaunchdTunService {
    static let label = "com.triglav.gui.tun"
    static let installedPlistPath = "/Library/LaunchDaemons/\(label).plist"

    static func stagedPlistURL(in directory: URL) -> URL {
        directory.appendingPathComponent("\(label).plist")
    }

    static func makePlist(binaryURL: URL, arguments: [String], logURL: URL) throws -> Data {
        let propertyList: [String: Any] = [
            "Label": label,
            "ProgramArguments": [binaryURL.path] + arguments,
            "RunAtLoad": true,
            "KeepAlive": false,
            "ProcessType": "Background",
            "WorkingDirectory": FileManager.default.homeDirectoryForCurrentUser.path,
            "StandardOutPath": logURL.path,
            "StandardErrorPath": logURL.path,
            "EnvironmentVariables": [
                "PATH": ProcessInfo.processInfo.environment["PATH"]
                    ?? "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            ],
        ]

        return try PropertyListSerialization.data(
            fromPropertyList: propertyList,
            format: .xml,
            options: 0
        )
    }

    static func install(from stagedPlistURL: URL) -> Result<Void, Error> {
        let command = [
            "launchctl bootout system/\(label) >/dev/null 2>&1 || true",
            "install -o root -g wheel -m 644 \(stagedPlistURL.path.shellEscaped) \(installedPlistPath.shellEscaped)",
            "launchctl bootstrap system \(installedPlistPath.shellEscaped)",
            "launchctl enable system/\(label)",
            "launchctl kickstart -k system/\(label)",
        ].joined(separator: "; ")

        switch ShellScriptRunner.runAppleScriptShell(command) {
        case .success:
            return .success(())
        case let .failure(error):
            return .failure(error)
        }
    }

    static func uninstall() -> Result<Void, Error> {
        let command = [
            "launchctl bootout system/\(label) >/dev/null 2>&1 || true",
            "rm -f \(installedPlistPath.shellEscaped)",
        ].joined(separator: "; ")

        switch ShellScriptRunner.runAppleScriptShell(command) {
        case .success:
            return .success(())
        case let .failure(error):
            return .failure(error)
        }
    }
}
