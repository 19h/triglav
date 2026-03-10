import Foundation
import ServiceManagement

struct HelperServiceSnapshot: Equatable {
    enum State: Equatable {
        case ready
        case approvalRequired
        case notRegistered
        case notFound
        case fallback
        case unknown
    }

    let state: State
    let title: String
    let detail: String

    var usesSMAppService: Bool {
        state != .fallback
    }

    var needsApproval: Bool {
        state == .approvalRequired
    }
}

enum PrivilegedHelperManager {
    static func isBundled(in bundle: Bundle = .main) -> Bool {
        bundledHelperURL(in: bundle) != nil && bundledHelperLaunchDaemonPlistURL(in: bundle) != nil
    }

    static func installHelperDaemon(completion: @escaping (Result<Void, Error>) -> Void) {
        let completionBox = CompletionBox(completion)

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                guard isBundled() else {
                    throw PrivilegedHelperError.helperNotBundled
                }

                try registerHelperDaemon()
                DispatchQueue.main.async {
                    completionBox.handler(.success(()))
                }
            } catch {
                DispatchQueue.main.async {
                    completionBox.handler(.failure(error))
                }
            }
        }
    }

    static func uninstallHelperDaemon(completion: @escaping (Result<Void, Error>) -> Void) {
        let completionBox = CompletionBox(completion)

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                guard isBundled() else {
                    throw PrivilegedHelperError.helperNotBundled
                }

                try unregisterHelperDaemon()
                DispatchQueue.main.async {
                    completionBox.handler(.success(()))
                }
            } catch {
                DispatchQueue.main.async {
                    completionBox.handler(.failure(error))
                }
            }
        }
    }

    static func installTunService(
        binaryURL: URL,
        arguments: [String],
        logURL: URL,
        completion: @escaping (Result<Void, Error>) -> Void
    ) {
        let completionBox = CompletionBox(completion)

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                guard let _ = bundledHelperURL(), bundledHelperLaunchDaemonPlistURL() != nil else {
                    throw PrivilegedHelperError.helperNotBundled
                }

                try registerHelperDaemon()
                let connectionBox = XPCConnectionBox(makeConnection())
                let proxy = remoteProxy(connectionBox: connectionBox, completionBox: completionBox)

                proxy.installTunService(
                    binarySourcePath: binaryURL.path,
                    arguments: arguments,
                    logPath: logURL.path
                ) { errorMessage in
                    DispatchQueue.main.async {
                        connectionBox.connection.invalidate()
                        if let errorMessage, !errorMessage.isEmpty {
                            completionBox.handler(.failure(PrivilegedHelperError.remote(errorMessage)))
                        } else {
                            completionBox.handler(.success(()))
                        }
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    completionBox.handler(.failure(error))
                }
            }
        }
    }

    static func uninstallTunService(completion: @escaping (Result<Void, Error>) -> Void) {
        let completionBox = CompletionBox(completion)

        DispatchQueue.global(qos: .userInitiated).async {
            let connectionBox = XPCConnectionBox(makeConnection())
            let proxy = remoteProxy(connectionBox: connectionBox, completionBox: completionBox)

            proxy.uninstallTunService { errorMessage in
                DispatchQueue.main.async {
                    connectionBox.connection.invalidate()
                    if let errorMessage, !errorMessage.isEmpty {
                        completionBox.handler(.failure(PrivilegedHelperError.remote(errorMessage)))
                    } else {
                        completionBox.handler(.success(()))
                    }
                }
            }
        }
    }

    static func bundledHelperURL(in bundle: Bundle = .main) -> URL? {
        let candidate = bundle.bundleURL
            .appendingPathComponent("Contents/Library/LaunchServices", isDirectory: true)
            .appendingPathComponent(PrivilegedHelperConstants.helperExecutableName)

        return FileManager.default.isExecutableFile(atPath: candidate.path) ? candidate : nil
    }

    static func bundledHelperLaunchDaemonPlistURL(in bundle: Bundle = .main) -> URL? {
        let candidate = bundle.bundleURL
            .appendingPathComponent("Contents/Library/LaunchDaemons", isDirectory: true)
            .appendingPathComponent(PrivilegedHelperConstants.helperLaunchDaemonPlistName)

        return FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
    }

    static func helperStatusSnapshot(in bundle: Bundle = .main) -> HelperServiceSnapshot {
        guard isBundled(in: bundle) else {
            return HelperServiceSnapshot(
                state: .fallback,
                title: "Admin prompt fallback",
                detail: "This build does not include the embedded daemon bundle, so TUN mode falls back to the direct administrator prompt flow."
            )
        }

        let service = helperService()
        return switch service.status {
        case .enabled:
            HelperServiceSnapshot(
                state: .ready,
                title: "Ready",
                detail: "The embedded root helper is registered and can launch the TUN daemon directly from the app bundle."
            )
        case .requiresApproval:
            HelperServiceSnapshot(
                state: .approvalRequired,
                title: "Approval required",
                detail: "Allow the Triglav background item in Login Items & Extensions, then retry the TUN connection."
            )
        case .notRegistered:
            HelperServiceSnapshot(
                state: .notRegistered,
                title: "Not registered yet",
                detail: "The helper is bundled correctly. The first TUN connection attempt will register it and may prompt for admin approval."
            )
        case .notFound:
            HelperServiceSnapshot(
                state: .notFound,
                title: "Helper not found",
                detail: "macOS could not find the embedded launch daemon metadata. Rebuild or reinstall the app bundle."
            )
        @unknown default:
            HelperServiceSnapshot(
                state: .unknown,
                title: "Unknown helper state",
                detail: "The Service Management framework returned an unknown helper status."
            )
        }
    }

    static func openLoginItemsSettings() {
        SMAppService.openSystemSettingsLoginItems()
    }

    private static func remoteProxy(
        connectionBox: XPCConnectionBox,
        completionBox: CompletionBox
    ) -> TriglavPrivilegedHelperXPCProtocol {
        connectionBox.connection.remoteObjectInterface = NSXPCInterface(with: TriglavPrivilegedHelperXPCProtocol.self)
        connectionBox.connection.resume()

        guard let proxy = connectionBox.connection.remoteObjectProxyWithErrorHandler({ error in
            DispatchQueue.main.async {
                connectionBox.connection.invalidate()
                completionBox.handler(.failure(error))
            }
        }) as? TriglavPrivilegedHelperXPCProtocol else {
            fatalError("Failed to create the privileged helper proxy")
        }

        return proxy
    }

    private static func makeConnection() -> NSXPCConnection {
        NSXPCConnection(machServiceName: PrivilegedHelperConstants.helperLabel)
    }

    private static func helperService() -> SMAppService {
        SMAppService.daemon(plistName: PrivilegedHelperConstants.helperLaunchDaemonPlistName)
    }

    private static func registerHelperDaemon() throws {
        let service = helperService()

        switch service.status {
        case .enabled:
            return
        case .requiresApproval:
            SMAppService.openSystemSettingsLoginItems()
            throw PrivilegedHelperError.requiresApproval
        case .notRegistered, .notFound:
            try service.register()
            if service.status == .requiresApproval {
                SMAppService.openSystemSettingsLoginItems()
                throw PrivilegedHelperError.requiresApproval
            }
        @unknown default:
            try service.register()
        }
    }

    private static func unregisterHelperDaemon() throws {
        let service = helperService()

        switch service.status {
        case .enabled, .requiresApproval:
            try service.unregister()
        case .notRegistered, .notFound:
            return
        @unknown default:
            try service.unregister()
        }
    }
}

private final class XPCConnectionBox: @unchecked Sendable {
    let connection: NSXPCConnection

    init(_ connection: NSXPCConnection) {
        self.connection = connection
    }
}

private final class CompletionBox: @unchecked Sendable {
    let handler: (Result<Void, Error>) -> Void

    init(_ handler: @escaping (Result<Void, Error>) -> Void) {
        self.handler = handler
    }
}

private enum PrivilegedHelperError: LocalizedError {
    case helperNotBundled
    case requiresApproval
    case remote(String)

    var errorDescription: String? {
        switch self {
        case .helperNotBundled:
            "The privileged helper is not embedded in this app build. Open the Xcode project and build the app target to include it."
        case .requiresApproval:
            "Approve the Triglav background helper in Login Items & Extensions, then try again."
        case let .remote(message):
            message
        }
    }
}
