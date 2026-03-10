import Foundation

final class PrivilegedHelperService: NSObject, NSXPCListenerDelegate, TriglavPrivilegedHelperXPCProtocol {
    private let fileManager = FileManager.default

    func listener(_ listener: NSXPCListener, shouldAcceptNewConnection newConnection: NSXPCConnection) -> Bool {
        newConnection.exportedInterface = NSXPCInterface(with: TriglavPrivilegedHelperXPCProtocol.self)
        newConnection.exportedObject = self
        newConnection.resume()
        return true
    }

    func ping(_ reply: @escaping (String) -> Void) {
        reply("pong")
    }

    func installTunService(
        binarySourcePath: String,
        arguments: [String],
        logPath: String,
        withReply reply: @escaping (String?) -> Void
    ) {
        do {
            try installTunService(binarySourcePath: binarySourcePath, arguments: arguments, logPath: logPath)
            reply(nil)
        } catch {
            reply(error.localizedDescription)
        }
    }

    func uninstallTunService(withReply reply: @escaping (String?) -> Void) {
        do {
            try uninstallTunService()
            reply(nil)
        } catch {
            reply(error.localizedDescription)
        }
    }

    private func installTunService(binarySourcePath: String, arguments: [String], logPath: String) throws {
        let sourceURL = URL(fileURLWithPath: binarySourcePath).standardizedFileURL
        guard fileManager.isExecutableFile(atPath: sourceURL.path) else {
            throw PrivilegedHelperToolError.invalidBinary(sourceURL.path)
        }

        let installedBinaryURL = URL(fileURLWithPath: PrivilegedHelperConstants.installedTriglavBinaryPath)
        let plistURL = URL(fileURLWithPath: PrivilegedHelperConstants.installedLaunchDaemonPlistPath)
        let logURL = URL(fileURLWithPath: logPath).standardizedFileURL

        try ensureDirectoryExists(at: installedBinaryURL.deletingLastPathComponent())
        try ensureDirectoryExists(at: logURL.deletingLastPathComponent())

        if fileManager.fileExists(atPath: installedBinaryURL.path) {
            try fileManager.removeItem(at: installedBinaryURL)
        }

        try fileManager.copyItem(at: sourceURL, to: installedBinaryURL)
        try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: installedBinaryURL.path)

        if !fileManager.fileExists(atPath: logURL.path) {
            fileManager.createFile(atPath: logURL.path, contents: nil)
        }
        try fileManager.setAttributes([.posixPermissions: 0o644], ofItemAtPath: logURL.path)

        _ = try? runTool(
            launchctlURL,
            arguments: ["bootout", "system/\(PrivilegedHelperConstants.tunServiceLabel)"]
        )

        if fileManager.fileExists(atPath: plistURL.path) {
            try fileManager.removeItem(at: plistURL)
        }

        let plistData = try makeLaunchDaemonPlist(
            binaryPath: installedBinaryURL.path,
            arguments: arguments,
            logPath: logURL.path
        )
        try plistData.write(to: plistURL, options: .atomic)
        try fileManager.setAttributes([.posixPermissions: 0o644], ofItemAtPath: plistURL.path)

        _ = try runTool(launchctlURL, arguments: ["bootstrap", "system", plistURL.path])
        _ = try? runTool(
            launchctlURL,
            arguments: ["enable", "system/\(PrivilegedHelperConstants.tunServiceLabel)"]
        )
        _ = try runTool(
            launchctlURL,
            arguments: ["kickstart", "-k", "system/\(PrivilegedHelperConstants.tunServiceLabel)"]
        )
    }

    private func uninstallTunService() throws {
        let installedBinaryURL = URL(fileURLWithPath: PrivilegedHelperConstants.installedTriglavBinaryPath)
        let plistURL = URL(fileURLWithPath: PrivilegedHelperConstants.installedLaunchDaemonPlistPath)

        _ = try? runTool(
            launchctlURL,
            arguments: ["bootout", "system/\(PrivilegedHelperConstants.tunServiceLabel)"]
        )

        if fileManager.fileExists(atPath: plistURL.path) {
            try fileManager.removeItem(at: plistURL)
        }

        if fileManager.fileExists(atPath: installedBinaryURL.path) {
            try fileManager.removeItem(at: installedBinaryURL)
        }
    }

    private func ensureDirectoryExists(at url: URL) throws {
        try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
        try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
    }

    private func makeLaunchDaemonPlist(binaryPath: String, arguments: [String], logPath: String) throws -> Data {
        let plist: [String: Any] = [
            "Label": PrivilegedHelperConstants.tunServiceLabel,
            "ProgramArguments": [binaryPath] + arguments,
            "RunAtLoad": true,
            "KeepAlive": false,
            "ProcessType": "Background",
            "StandardOutPath": logPath,
            "StandardErrorPath": logPath,
            "EnvironmentVariables": [
                "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            ],
        ]

        return try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0)
    }

    private func runTool(_ executableURL: URL, arguments: [String]) throws -> String {
        let process = Process()
        let pipe = Pipe()

        process.executableURL = executableURL
        process.arguments = arguments
        process.standardOutput = pipe
        process.standardError = pipe

        try process.run()
        process.waitUntilExit()

        let output = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard process.terminationStatus == 0 else {
            throw PrivilegedHelperToolError.commandFailed(
                executableURL.path,
                arguments.joined(separator: " "),
                output
            )
        }

        return output
    }

    private var launchctlURL: URL {
        URL(fileURLWithPath: "/bin/launchctl")
    }
}

private enum PrivilegedHelperToolError: LocalizedError {
    case invalidBinary(String)
    case commandFailed(String, String, String)

    var errorDescription: String? {
        switch self {
        case let .invalidBinary(path):
            return "The embedded Triglav CLI is missing or not executable at \(path)."
        case let .commandFailed(executable, arguments, output):
            if output.isEmpty {
                return "\(executable) \(arguments) failed."
            }
            return "\(executable) \(arguments) failed: \(output)"
        }
    }
}

@main
enum PrivilegedHelperMain {
    static func main() {
        let service = PrivilegedHelperService()
        let listener = NSXPCListener(machServiceName: PrivilegedHelperConstants.helperLabel)
        listener.delegate = service
        listener.resume()
        RunLoop.current.run()
    }
}
