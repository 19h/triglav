import AppKit
import Combine
import Darwin
import Foundation
import SystemConfiguration

enum ConnectionMode: String, Codable, CaseIterable, Identifiable {
    case tun
    case proxy

    var id: Self { self }

    var title: String {
        switch self {
        case .tun:
            "TUN VPN"
        case .proxy:
            "Proxy"
        }
    }

    var subtitle: String {
        switch self {
        case .tun:
            "Routes traffic through a privileged macOS tunnel session."
        case .proxy:
            "Runs the Triglav client as a local SOCKS or HTTP proxy."
        }
    }

    var command: String {
        switch self {
        case .tun:
            "tun"
        case .proxy:
            "connect"
        }
    }

    var requiresPrivileges: Bool {
        self == .tun
    }
}

enum SchedulingPreset: String, Codable, CaseIterable, Identifiable {
    case adaptive
    case wrr
    case latency
    case loss
    case redundant
    case primaryBackup

    var id: Self { self }

    var title: String {
        switch self {
        case .adaptive:
            "Adaptive"
        case .wrr:
            "WRR"
        case .latency:
            "Lowest latency"
        case .loss:
            "Lowest loss"
        case .redundant:
            "Redundant"
        case .primaryBackup:
            "Primary / backup"
        }
    }

    var cliValue: String {
        switch self {
        case .adaptive:
            "adaptive"
        case .wrr:
            "wrr"
        case .latency:
            "latency"
        case .loss:
            "loss"
        case .redundant:
            "redundant"
        case .primaryBackup:
            "primary-backup"
        }
    }
}

enum ConnectionState: String, Equatable {
    case disconnected
    case starting
    case connected
    case stopping
    case error

    var title: String {
        switch self {
        case .disconnected:
            "Disconnected"
        case .starting:
            "Starting"
        case .connected:
            "Connected"
        case .stopping:
            "Stopping"
        case .error:
            "Needs attention"
        }
    }

    var symbolName: String {
        switch self {
        case .disconnected:
            "bolt.horizontal.circle"
        case .starting:
            "arrow.triangle.2.circlepath.circle"
        case .connected:
            "bolt.horizontal.circle.fill"
        case .stopping:
            "pause.circle"
        case .error:
            "exclamationmark.triangle.fill"
        }
    }
}

struct NetworkInterface: Codable, Hashable, Identifiable {
    let name: String
    let displayName: String
    let kind: String

    var id: String { name }
}

struct AppSettings: Codable {
    var authKey = ""
    var mode: ConnectionMode = .tun
    var strategy: SchedulingPreset = .adaptive
    var autoDiscoverInterfaces = true
    var selectedInterfaces: [String] = []
    var fullTunnel = true
    var routes = ""
    var excludedRoutes = ""
    var tunName = "tg0"
    var tunnelIPv4 = "10.0.85.1"
    var socksPort = 1080
    var httpProxyPort = 0
    var binaryPath = ""
    var showStatusBarIcon = true
}

struct RuntimeStatus {
    var state: ConnectionState = .disconnected
    var startedAt: Date?
    var pid: Int32?
    var lastError: String?
    var lastCommand = ""
    var privilegedLaunch = false
    var logLines: [String] = []
}

final class AppModel: ObservableObject, @unchecked Sendable {
    private let sceneState: AppSceneState
    let menuBarState: MenuBarState

    @Published var settings: AppSettings {
        didSet {
            schedulePersistSettings()
            if oldValue.showStatusBarIcon != settings.showStatusBarIcon {
                sceneState.showStatusBarIcon = settings.showStatusBarIcon
                ActivationPolicyController.apply(showStatusBarIcon: settings.showStatusBarIcon)
            }
        }
    }

    @Published private(set) var availableInterfaces: [NetworkInterface]
    @Published private(set) var runtime = RuntimeStatus()
    @Published private(set) var liveStatus: ClientStatusSnapshot?
    @Published private(set) var helperStatus: HelperServiceSnapshot
    @Published private(set) var helperActionInProgress = false
    @Published private(set) var helperFeedbackMessage: String?

    let supportDirectoryURL: URL
    let settingsURL: URL
    let logsDirectoryURL: URL

    private var clientProcess: Process?
    private var privilegedMonitor: Timer?
    private var statusPollTimer: Timer?
    private var persistWorkItem: DispatchWorkItem?
    private var appDidBecomeActiveObserver: NSObjectProtocol?
    private var tunLogURL: URL?
    private var logOffset: UInt64 = 0
    private var streamRemainder = ""
    private var statusFailureCount = 0

    init(sceneState: AppSceneState, menuBarState: MenuBarState, fileManager: FileManager = .default) {
        self.sceneState = sceneState
        self.menuBarState = menuBarState

        let supportRoot = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support", isDirectory: true)
        let logsRoot = fileManager.urls(for: .libraryDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library", isDirectory: true)

        let appSupport = supportRoot.appendingPathComponent("TriglavGUI", isDirectory: true)
        let logs = logsRoot.appendingPathComponent("Logs/TriglavGUI", isDirectory: true)

        try? fileManager.createDirectory(at: appSupport, withIntermediateDirectories: true)
        try? fileManager.createDirectory(at: logs, withIntermediateDirectories: true)

        supportDirectoryURL = appSupport
        settingsURL = appSupport.appendingPathComponent("settings.json")
        logsDirectoryURL = logs

        settings = Self.loadSettings(from: settingsURL) ?? AppSettings()
        availableInterfaces = NetworkInterfaceProvider.listInterfaces()
        liveStatus = nil
        helperStatus = PrivilegedHelperManager.helperStatusSnapshot()
        tunLogURL = Self.latestTunLogURL(in: logs)

        normalizeSelectedInterfaces()
        sceneState.showStatusBarIcon = settings.showStatusBarIcon
        ActivationPolicyController.apply(showStatusBarIcon: settings.showStatusBarIcon)
        syncMenuBarState()
        appDidBecomeActiveObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.refreshHelperStatus()
        }
        refreshHelperStatus()
        startStatusPolling()
    }

    var menuBarSymbol: String {
        runtime.state.symbolName
    }

    var latestLogLine: String {
        runtime.logLines.last ?? ""
    }

    var statusHeadline: String {
        switch runtime.state {
        case .connected:
            if let quality = liveStatus?.quality {
                return "Connected over \(quality.usableUplinks)/\(quality.totalUplinks) uplinks"
            }
            if let startedAt = runtime.startedAt {
                return "Connected for \(Self.relativeDuration(since: startedAt))"
            }
            return "Connected"
        case .starting:
            return "Launching \(settings.mode.title.lowercased())"
        case .stopping:
            return "Stopping \(settings.mode.title.lowercased())"
        case .error:
            return runtime.lastError ?? "Triglav needs attention"
        case .disconnected:
            return settings.mode.subtitle
        }
    }

    var statusDetail: String {
        if let liveStatus {
            var parts: [String] = []

            if let processId = liveStatus.processId {
                parts.append("PID \(processId)")
            }

            if let sessionId = liveStatus.sessionId {
                parts.append("Session \(Self.shortIdentifier(sessionId))")
            }

            if let quality = liveStatus.quality {
                parts.append(String(format: "RTT %.1f ms", quality.avgRttMs))
                parts.append(String(format: "Loss %.1f%%", quality.avgLossPercent))
            }

            if !parts.isEmpty {
                return parts.joined(separator: " · ")
            }
        }

        if let pid = runtime.pid {
            return "PID \(pid) \u{00B7} \(interfaceSummary)"
        }

        if let lastError = runtime.lastError {
            return lastError
        }

        return interfaceSummary
    }

    var interfaceSummary: String {
        if settings.autoDiscoverInterfaces {
            return "Auto-discovering active uplinks"
        }

        if settings.selectedInterfaces.isEmpty {
            return "Using Triglav defaults when no interface is pinned"
        }

        return settings.selectedInterfaces.joined(separator: ", ")
    }

    var resolvedBinaryDisplay: String {
        resolvedBinaryURL()?.path ?? "Auto-detect `triglav` from PATH or Cargo outputs"
    }

    var activeLogLocation: String {
        tunLogURL?.path ?? logsDirectoryURL.path
    }

    var statusEndpointDisplay: String {
        ClientStatusEndpoint.statusURL.absoluteString
    }

    var liveUplinks: [ClientUplinkSnapshot] {
        liveStatus?.uplinks ?? []
    }

    var trafficSummary: String {
        guard let liveStatus else { return "Awaiting client status endpoint" }
        return "TX \(liveStatus.totalBytesSent.byteCountDisplay) · RX \(liveStatus.totalBytesReceived.byteCountDisplay)"
    }

    var sessionDisplay: String {
        liveStatus?.sessionId ?? "Not available"
    }

    var connectionDisplay: String {
        liveStatus?.connectionId ?? "Not available"
    }

    var qualityDisplay: String {
        guard let quality = liveStatus?.quality else { return "No live quality snapshot yet" }
        return String(
            format: "%.1f ms RTT · %.1f%% loss · %.1f Mbps aggregate",
            quality.avgRttMs,
            quality.avgLossPercent,
            quality.totalBandwidthMbps
        )
    }

    var tunnelDisplay: String {
        guard let tunnel = liveStatus?.tunnel else { return settings.mode == .tun ? settings.tunName : "Not in TUN mode" }
        return tunnel.fullTunnel ? "\(tunnel.tunName) · full tunnel" : "\(tunnel.tunName) · split tunnel"
    }

    var commandPreview: String {
        buildCommandPreview(binaryPath: resolvedBinaryURL()?.path ?? binaryCandidateDisplayName)
    }

    var helperStatusTitle: String {
        helperStatus.title
    }

    var helperStatusDetail: String {
        helperStatus.detail
    }

    var helperStatusDisplayDetail: String {
        helperFeedbackMessage ?? helperStatus.detail
    }

    var helperNeedsApproval: Bool {
        helperStatus.needsApproval
    }

    var helperUsesSMAppService: Bool {
        helperStatus.usesSMAppService
    }

    var helperNeedsInstall: Bool {
        guard helperUsesSMAppService else { return false }
        return switch helperStatus.state {
        case .ready, .approvalRequired:
            false
        case .notRegistered, .notFound, .unknown, .fallback:
            true
        }
    }

    var helperCanUninstall: Bool {
        return helperUsesSMAppService && !helperNeedsInstall
    }

    var canConnect: Bool {
        switch runtime.state {
        case .disconnected, .error:
            true
        case .starting, .connected, .stopping:
            false
        }
    }

    var canDisconnect: Bool {
        switch runtime.state {
        case .starting, .connected, .stopping:
            true
        case .disconnected, .error:
            false
        }
    }

    func refreshInterfaces() {
        availableInterfaces = NetworkInterfaceProvider.listInterfaces()
        normalizeSelectedInterfaces()
    }

    func toggleInterface(_ interfaceName: String) {
        if settings.selectedInterfaces.contains(interfaceName) {
            settings.selectedInterfaces.removeAll { $0 == interfaceName }
        } else {
            settings.selectedInterfaces.append(interfaceName)
            settings.selectedInterfaces.sort()
        }
    }

    func clearLogs() {
        runtime.logLines = []
    }

    func copyCommandToPasteboard() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(commandPreview, forType: .string)
        appendSystemLog("Copied the launch command preview to the clipboard.")
    }

    func openLogsFolder() {
        NSWorkspace.shared.open(logsDirectoryURL)
    }

    func refreshHelperStatus() {
        let snapshot = PrivilegedHelperManager.helperStatusSnapshot()
        if helperStatus != snapshot {
            helperStatus = snapshot
        }

        if helperFeedbackMessage != nil,
           snapshot.state == .ready || snapshot.state == .notRegistered || snapshot.state == .approvalRequired
        {
            helperFeedbackMessage = nil
        }
    }

    func openHelperSettings() {
        PrivilegedHelperManager.openLoginItemsSettings()
    }

    func installHelper() {
        guard helperUsesSMAppService else { return }
        guard !helperActionInProgress else { return }

        helperActionInProgress = true
        helperFeedbackMessage = nil

        PrivilegedHelperManager.installHelperDaemon { [weak self] result in
            guard let self else { return }

            self.helperActionInProgress = false
            self.refreshHelperStatus()

            switch result {
            case .success:
                if self.helperNeedsApproval {
                    self.helperFeedbackMessage = "Helper registered. Approve it in Login Items & Extensions to enable privileged TUN launches."
                    self.openHelperSettings()
                } else {
                    self.helperFeedbackMessage = "Helper installed and ready for privileged TUN launches."
                }
                self.appendSystemLog("Helper registration completed.")
            case let .failure(error):
                self.helperFeedbackMessage = error.localizedDescription
                self.appendSystemLog("Helper install failed: \(error.localizedDescription)")
            }
        }
    }

    func uninstallHelper() {
        guard helperUsesSMAppService else { return }
        guard !helperActionInProgress else { return }

        if runtime.privilegedLaunch && canDisconnect {
            helperFeedbackMessage = "Disconnect the privileged tunnel before uninstalling the helper."
            return
        }

        helperActionInProgress = true
        helperFeedbackMessage = nil

        PrivilegedHelperManager.uninstallHelperDaemon { [weak self] result in
            guard let self else { return }

            self.helperActionInProgress = false
            self.refreshHelperStatus()

            switch result {
            case .success:
                self.helperFeedbackMessage = "Helper uninstalled. Future TUN launches will need the helper installed again."
                self.appendSystemLog("Helper uninstalled.")
            case let .failure(error):
                self.helperFeedbackMessage = error.localizedDescription
                self.appendSystemLog("Helper uninstall failed: \(error.localizedDescription)")
            }
        }
    }

    func connect() {
        guard canConnect else { return }

        if settings.mode.requiresPrivileges {
            refreshHelperStatus()
            if helperNeedsApproval {
                openHelperSettings()
                presentError("Approve the Triglav helper in Login Items & Extensions, then reconnect.")
                return
            }
        }

        guard !settings.authKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            presentError("Paste a tg1_ auth key before connecting.")
            return
        }

        if settings.mode == .tun,
           !settings.autoDiscoverInterfaces,
           settings.selectedInterfaces.isEmpty
        {
            presentError("Pick at least one interface or turn auto-discovery back on for TUN mode.")
            return
        }

        guard let binaryURL = resolvedBinaryURL() else {
            presentError("Could not find the triglav binary. Set it explicitly in Settings.")
            return
        }

        runtime = RuntimeStatus(
            state: .starting,
            startedAt: Date(),
            pid: nil,
            lastError: nil,
            lastCommand: buildCommandPreview(binaryPath: binaryURL.path),
            privilegedLaunch: settings.mode.requiresPrivileges,
            logLines: []
        )
        liveStatus = nil
        statusFailureCount = 0
        streamRemainder = ""
        syncMenuBarState()

        appendSystemLog("Launching \(settings.mode.title.lowercased())...")

        let arguments = buildArguments()

        if settings.mode.requiresPrivileges {
            startPrivilegedTun(binaryURL: binaryURL, arguments: arguments)
        } else {
            startManagedProcess(binaryURL: binaryURL, arguments: arguments)
        }
    }

    func disconnect() {
        guard canDisconnect else { return }

        runtime.state = .stopping
        syncMenuBarState()
        appendSystemLog("Disconnecting...")

        if settings.mode.requiresPrivileges {
            if PrivilegedHelperManager.isBundled() {
                PrivilegedHelperManager.uninstallTunService { [weak self] result in
                    guard let self else { return }

                switch result {
                case .success:
                    self.refreshHelperStatus()
                    self.finalizeDisconnect(message: "Stopped privileged helper tunnel service.")
                case let .failure(error):
                    self.refreshHelperStatus()
                    self.presentError("Failed to stop privileged helper tunnel service: \(error.localizedDescription)")
                }
            }

                return
            }

            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                let result = LaunchdTunService.uninstall()
                DispatchQueue.main.async {
                    guard let self else { return }

                    switch result {
                    case .success:
                        self.finalizeDisconnect(message: "Stopped launchd tunnel service.")
                    case let .failure(error):
                        self.presentError("Failed to stop launchd tunnel service: \(error.localizedDescription)")
                    }
                }
            }

            return
        }

        if let clientProcess, clientProcess.isRunning {
            clientProcess.terminate()
        } else {
            finalizeDisconnect(message: "Disconnected.")
        }
    }

    func quitApp() {
        if canDisconnect {
            disconnect()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                NSApplication.shared.terminate(nil)
            }
        } else {
            DispatchQueue.main.async {
                NSApplication.shared.terminate(nil)
            }
        }
    }

    private func buildArguments() -> [String] {
        var arguments = [settings.mode.command, settings.authKey.trimmingCharacters(in: .whitespacesAndNewlines)]

        switch settings.mode {
        case .tun:
            arguments.append(contentsOf: [
                "--metrics-addr", ClientStatusEndpoint.address,
                "--strategy", settings.strategy.cliValue,
                "--tun-name", settings.tunName.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty ?? "tg0",
                "--ipv4", settings.tunnelIPv4.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty ?? "10.0.85.1",
                "--verbose",
            ])

            if settings.fullTunnel {
                arguments.append("--full-tunnel")
            }

            if settings.autoDiscoverInterfaces {
                arguments.append("--auto-discover")
            } else {
                for interfaceName in settings.selectedInterfaces {
                    arguments.append(contentsOf: ["-i", interfaceName])
                }
            }

            for route in tokenizeList(settings.routes) {
                arguments.append(contentsOf: ["--route", route])
            }

            for route in tokenizeList(settings.excludedRoutes) {
                arguments.append(contentsOf: ["--exclude", route])
            }

        case .proxy:
            arguments.append(contentsOf: [
                "--foreground",
                "--metrics-addr", ClientStatusEndpoint.address,
                "--strategy", settings.strategy.cliValue,
                "--verbose",
            ])

            if settings.autoDiscoverInterfaces {
                arguments.append("--auto-discover")
            } else {
                for interfaceName in settings.selectedInterfaces {
                    arguments.append(contentsOf: ["-i", interfaceName])
                }
            }

            if settings.socksPort > 0 {
                arguments.append(contentsOf: ["--socks", String(settings.socksPort)])
            }

            if settings.httpProxyPort > 0 {
                arguments.append(contentsOf: ["--http-proxy", String(settings.httpProxyPort)])
            }
        }

        return arguments
    }

    private func buildCommandPreview(binaryPath: String) -> String {
        let command = [binaryPath] + buildArguments()
        return command.map(\.shellEscaped).joined(separator: " ")
    }

    private func startManagedProcess(binaryURL: URL, arguments: [String]) {
        let process = Process()
        let pipe = Pipe()

        process.executableURL = binaryURL
        process.arguments = arguments
        process.standardOutput = pipe
        process.standardError = pipe

        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }

            let text = String(decoding: data, as: UTF8.self)
            DispatchQueue.main.async {
                self?.consumeProcessOutput(text)
            }
        }

        process.terminationHandler = { [weak self] terminatedProcess in
            DispatchQueue.main.async {
                self?.handleManagedProcessExit(process: terminatedProcess)
            }
        }

        do {
            try process.run()
            clientProcess = process
            runtime.pid = process.processIdentifier
            appendSystemLog("Client started as pid \(process.processIdentifier).")
        } catch {
            pipe.fileHandleForReading.readabilityHandler = nil
            presentError("Failed to launch triglav: \(error.localizedDescription)")
        }
    }

    private func startPrivilegedTun(binaryURL: URL, arguments: [String]) {
        let logURL = logsDirectoryURL.appendingPathComponent("tun-\(Self.logTimestamp()).log")
        tunLogURL = logURL
        logOffset = 0
        _ = FileManager.default.createFile(atPath: logURL.path, contents: nil)

        if PrivilegedHelperManager.isBundled() {
            PrivilegedHelperManager.installTunService(
                binaryURL: binaryURL,
                arguments: arguments,
                logURL: logURL
            ) { [weak self] result in
                guard let self else { return }

                switch result {
                case .success:
                    self.refreshHelperStatus()
                    self.appendSystemLog("Blessed helper installed and started the tunnel service.")
                    self.startPrivilegedMonitor()
                case let .failure(error):
                    self.refreshHelperStatus()
                    self.presentError("Failed to start privileged helper tunnel service: \(error.localizedDescription)")
                }
            }
            return
        }

        let stagedPlistURL = LaunchdTunService.stagedPlistURL(in: supportDirectoryURL)

        do {
            let plistData = try LaunchdTunService.makePlist(
                binaryURL: binaryURL,
                arguments: arguments,
                logURL: logURL
            )
            try plistData.write(to: stagedPlistURL, options: .atomic)
        } catch {
            presentError("Failed to prepare the launchd tunnel service: \(error.localizedDescription)")
            return
        }

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = LaunchdTunService.install(from: stagedPlistURL)
            DispatchQueue.main.async {
                guard let self else { return }

                switch result {
                case .success:
                    self.appendSystemLog("Installed and started the launchd tunnel service.")
                    self.startPrivilegedMonitor()
                case let .failure(error):
                    self.presentError("Failed to start the launchd tunnel service: \(error.localizedDescription)")
                }
            }
        }
    }

    private func startPrivilegedMonitor() {
        privilegedMonitor?.invalidate()

        privilegedMonitor = Timer.scheduledTimer(withTimeInterval: 0.8, repeats: true) { [weak self] _ in
            self?.readPrivilegedLogUpdates()
        }
        privilegedMonitor?.tolerance = 0.2
        readPrivilegedLogUpdates()
    }

    private func startStatusPolling() {
        statusPollTimer?.invalidate()

        statusPollTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.pollStatusEndpoint()
        }
        statusPollTimer?.tolerance = 0.2
        pollStatusEndpoint()
    }

    private func pollStatusEndpoint() {
        var request = URLRequest(url: ClientStatusEndpoint.statusURL)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.timeoutInterval = 0.8

        URLSession.shared.dataTask(with: request) { [weak self] data, response, _ in
            guard let self else { return }

            guard let response = response as? HTTPURLResponse,
                  response.statusCode == 200,
                  let data,
                  let snapshot = try? JSONDecoder.triglavStatusDecoder.decode(ClientStatusSnapshot.self, from: data),
                  snapshot.isClientRole
            else {
                DispatchQueue.main.async {
                    self.handleStatusUnavailable()
                }
                return
            }

            DispatchQueue.main.async {
                self.handleLiveStatus(snapshot)
            }
        }.resume()
    }

    private func handleLiveStatus(_ snapshot: ClientStatusSnapshot) {
        if liveStatus != snapshot {
            liveStatus = snapshot
        }
        statusFailureCount = 0

        if let processId = snapshot.processId,
           runtime.pid != Int32(processId)
        {
            runtime.pid = Int32(processId)
        }

        if snapshot.uptimeSeconds > 0 {
            let startedAt = Date(timeIntervalSinceNow: -TimeInterval(snapshot.uptimeSeconds))
            if let currentStartedAt = runtime.startedAt {
                if abs(currentStartedAt.timeIntervalSince(startedAt)) > 1.0 {
                    runtime.startedAt = startedAt
                }
            } else {
                runtime.startedAt = startedAt
            }
        }

        let nextState = Self.connectionState(from: snapshot.state)
        if runtime.state != .stopping {
            if runtime.state != nextState {
                runtime.state = nextState
            }
        }

        if runtime.state == .connected,
           runtime.lastError != nil
        {
            runtime.lastError = nil
        }

        syncMenuBarState()

        if snapshot.mode == ConnectionMode.tun.rawValue {
            tunLogURL = tunLogURL ?? Self.latestTunLogURL(in: logsDirectoryURL)
            if privilegedMonitor == nil {
                startPrivilegedMonitor()
            }
        }
    }

    private func handleStatusUnavailable() {
        statusFailureCount += 1

        guard statusFailureCount >= 3 else { return }

        if runtime.state == .stopping {
            liveStatus = nil
            return
        }

        if liveStatus != nil {
            liveStatus = nil

            if clientProcess == nil,
               settings.mode == .tun,
               runtime.state == .connected
            {
                let lastMeaningfulLine = runtime.logLines.reversed().first { !$0.isEmpty && !$0.hasPrefix("[gui]") }
                if let lastMeaningfulLine {
                    presentError(lastMeaningfulLine)
                } else {
                    runtime.state = .disconnected
                    runtime.pid = nil
                }
                syncMenuBarState()
            }
        }
    }

    private func readPrivilegedLogUpdates() {
        guard let tunLogURL,
              let handle = try? FileHandle(forReadingFrom: tunLogURL)
        else {
            return
        }

        defer {
            try? handle.close()
        }

        do {
            try handle.seek(toOffset: logOffset)
            let data = try handle.readToEnd() ?? Data()

            guard !data.isEmpty else { return }

            logOffset += UInt64(data.count)
            let text = String(decoding: data, as: UTF8.self)
            consumeProcessOutput(text)
        } catch {
            appendSystemLog("Log tailing failed: \(error.localizedDescription)")
        }
    }

    private func consumeProcessOutput(_ text: String) {
        let combined = streamRemainder + text
        let lines = combined.components(separatedBy: .newlines)

        if combined.hasSuffix("\n") || combined.hasSuffix("\r") {
            streamRemainder = ""
            appendLogLines(lines)
        } else {
            streamRemainder = lines.last ?? ""
            appendLogLines(Array(lines.dropLast()))
        }
    }

    private func appendLogLines(_ lines: [String]) {
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }

            runtime.logLines.append(trimmed)
            updateState(from: trimmed)
        }

        if runtime.logLines.count > 300 {
            runtime.logLines.removeFirst(runtime.logLines.count - 300)
        }
    }

    private func appendSystemLog(_ line: String) {
        runtime.logLines.append("[gui] \(line)")

        if runtime.logLines.count > 300 {
            runtime.logLines.removeFirst(runtime.logLines.count - 300)
        }
    }

    private func updateState(from line: String) {
        let lowercaseLine = line.lowercased()

        if lowercaseLine.contains("connected!")
            || lowercaseLine.contains("connected. press ctrl+c to disconnect")
            || lowercaseLine.contains("tunnel running")
        {
            runtime.state = .connected
            runtime.lastError = nil
            return
        }

        if lowercaseLine.contains("disconnecting") || lowercaseLine.contains("shutting down") {
            runtime.state = .stopping
            return
        }

        if lowercaseLine.contains("connection failed")
            || lowercaseLine.contains("tunnel error")
            || lowercaseLine.contains("failed to")
            || lowercaseLine.hasPrefix("error:")
        {
            runtime.state = .error
            runtime.lastError = line
            return
        }

        if lowercaseLine.contains("tunnel stopped") || lowercaseLine.contains("disconnected.") {
            finalizeDisconnect(message: line)
        }
    }

    private func handleManagedProcessExit(process: Process) {
        clientProcess = nil

        if let pipe = process.standardOutput as? Pipe {
            pipe.fileHandleForReading.readabilityHandler = nil
        }

        if runtime.state == .stopping {
            finalizeDisconnect(message: "Disconnected.")
            return
        }

        if process.terminationStatus == 0 {
            finalizeDisconnect(message: "Triglav exited cleanly.")
        } else {
            presentError("Triglav exited with status \(process.terminationStatus).")
        }
    }

    private func finalizeDisconnect(message: String) {
        privilegedMonitor?.invalidate()
        privilegedMonitor = nil
        clientProcess = nil
        liveStatus = nil
        statusFailureCount = 0
        runtime.state = .disconnected
        runtime.pid = nil
        runtime.lastError = nil
        runtime.privilegedLaunch = false
        syncMenuBarState()
        appendSystemLog(message)
    }

    private func presentError(_ message: String) {
        privilegedMonitor?.invalidate()
        privilegedMonitor = nil
        clientProcess = nil
        liveStatus = nil
        runtime.state = .error
        runtime.pid = nil
        runtime.lastError = message
        syncMenuBarState()
        appendSystemLog(message)
    }

    private func syncMenuBarState() {
        menuBarState.update(
            connectionState: runtime.state,
            modeTitle: settings.mode.title
        )
    }

    private func persistSettings() {
        guard let data = try? JSONEncoder().encode(settings) else { return }
        try? data.write(to: settingsURL, options: .atomic)
    }

    private func schedulePersistSettings() {
        persistWorkItem?.cancel()

        let workItem = DispatchWorkItem { [weak self] in
            self?.persistSettings()
        }
        persistWorkItem = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.25, execute: workItem)
    }

    private func normalizeSelectedInterfaces() {
        let available = Set(availableInterfaces.map(\.name))
        let filtered = settings.selectedInterfaces.filter { available.contains($0) }
        if filtered != settings.selectedInterfaces {
            settings.selectedInterfaces = filtered
        }
    }

    private func resolvedBinaryURL() -> URL? {
        let fileManager = FileManager.default
        let explicitPath = settings.binaryPath.trimmingCharacters(in: .whitespacesAndNewlines)

        if let explicitURL = resolveBinaryCandidate(explicitPath, fileManager: fileManager) {
            return explicitURL
        }

        if let bundledResources = Bundle.main.resourceURL {
            let bundledBinary = bundledResources.appendingPathComponent("bin/triglav")
            if fileManager.isExecutableFile(atPath: bundledBinary.path) {
                return bundledBinary
            }
        }

        let pathBinary = resolvePathBinary(named: "triglav", fileManager: fileManager)
        if let pathBinary {
            return pathBinary
        }

        let cwd = URL(fileURLWithPath: fileManager.currentDirectoryPath)
        let localCandidates = [
            cwd.appendingPathComponent("target/debug/triglav"),
            cwd.appendingPathComponent("target/release/triglav"),
            cwd.appendingPathComponent("../target/debug/triglav"),
            cwd.appendingPathComponent("../target/release/triglav"),
            cwd.appendingPathComponent("../../target/debug/triglav"),
            cwd.appendingPathComponent("../../target/release/triglav"),
        ]

        for candidate in localCandidates where fileManager.isExecutableFile(atPath: candidate.standardizedFileURL.path) {
            return candidate.standardizedFileURL
        }

        return nil
    }

    private func resolveBinaryCandidate(_ value: String, fileManager: FileManager) -> URL? {
        guard !value.isEmpty else { return nil }

        if value.contains("/") || value.hasPrefix("~") {
            let expanded = NSString(string: value).expandingTildeInPath
            if fileManager.isExecutableFile(atPath: expanded) {
                return URL(fileURLWithPath: expanded)
            }
            return nil
        }

        return resolvePathBinary(named: value, fileManager: fileManager)
    }

    private func resolvePathBinary(named binaryName: String, fileManager: FileManager) -> URL? {
        let searchPath = ProcessInfo.processInfo.environment["PATH"]
            ?? "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

        for directory in searchPath.split(separator: ":") {
            let url = URL(fileURLWithPath: String(directory)).appendingPathComponent(binaryName)
            if fileManager.isExecutableFile(atPath: url.path) {
                return url
            }
        }

        return nil
    }

    private var binaryCandidateDisplayName: String {
        let explicitPath = settings.binaryPath.trimmingCharacters(in: .whitespacesAndNewlines)
        return explicitPath.isEmpty ? "triglav" : explicitPath
    }

    private static func loadSettings(from url: URL) -> AppSettings? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(AppSettings.self, from: data)
    }

    private static func latestTunLogURL(in directory: URL) -> URL? {
        let fileManager = FileManager.default
        guard let candidates = try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        return candidates
            .filter { $0.lastPathComponent.hasPrefix("tun-") && $0.pathExtension == "log" }
            .sorted {
                let lhsDate = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let rhsDate = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return lhsDate > rhsDate
            }
            .first
    }

    private static func connectionState(from rawValue: String) -> ConnectionState {
        switch rawValue.lowercased() {
        case "running", "connected":
            .connected
        case "starting", "connecting", "handshaking":
            .starting
        case "disconnecting":
            .stopping
        case "failed":
            .error
        default:
            .disconnected
        }
    }

    private static func shortIdentifier(_ value: String) -> String {
        String(value.prefix(8))
    }

    private static func processExists(pid: Int32) -> Bool {
        if kill(pid, 0) == 0 {
            return true
        }

        return errno == EPERM
    }

    private static func relativeDuration(since date: Date) -> String {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.hour, .minute, .second]
        formatter.unitsStyle = .abbreviated
        formatter.maximumUnitCount = 2
        return formatter.string(from: date, to: Date()) ?? "moments"
    }

    private static func logTimestamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: Date())
    }

    private func tokenizeList(_ text: String) -> [String] {
        text.split(whereSeparator: { $0 == "," || $0 == "\n" || $0 == ";" })
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    deinit {
        privilegedMonitor?.invalidate()
        statusPollTimer?.invalidate()
        persistWorkItem?.cancel()
        if let appDidBecomeActiveObserver {
            NotificationCenter.default.removeObserver(appDidBecomeActiveObserver)
        }
    }
}

private enum ActivationPolicyController {
    static func apply(showStatusBarIcon _: Bool) {
        DispatchQueue.main.async {
            NSApplication.shared.setActivationPolicy(.regular)
        }
    }
}

private enum NetworkInterfaceProvider {
    static func listInterfaces() -> [NetworkInterface] {
        guard let interfaces = SCNetworkInterfaceCopyAll() as? [SCNetworkInterface] else {
            return []
        }

        let mapped = interfaces.compactMap { interface -> NetworkInterface? in
            guard let bsdName = SCNetworkInterfaceGetBSDName(interface) as String? else {
                return nil
            }

            let displayName = (SCNetworkInterfaceGetLocalizedDisplayName(interface) as String?) ?? bsdName
            let kind = readableInterfaceKind(SCNetworkInterfaceGetInterfaceType(interface) as String?)

            return NetworkInterface(name: bsdName, displayName: displayName, kind: kind)
        }

        return Array(Set(mapped)).sorted { lhs, rhs in
            lhs.name.localizedStandardCompare(rhs.name) == .orderedAscending
        }
    }

    private static func readableInterfaceKind(_ rawType: String?) -> String {
        let wifi = kSCNetworkInterfaceTypeIEEE80211 as String
        let ethernet = kSCNetworkInterfaceTypeEthernet as String
        let wwan = kSCNetworkInterfaceTypeWWAN as String
        let bluetooth = kSCNetworkInterfaceTypeBluetooth as String
        let firewire = kSCNetworkInterfaceTypeFireWire as String

        return switch rawType {
        case wifi:
            "Wi-Fi"
        case ethernet:
            "Ethernet"
        case wwan:
            "Cellular"
        case bluetooth:
            "Bluetooth"
        case firewire:
            "FireWire"
        case "VPN":
            "VPN"
        default:
            rawType ?? "Network"
        }
    }
}

enum ShellScriptRunner {
    static func runAppleScriptShell(_ command: String) -> Result<String, Error> {
        let process = Process()
        let stdout = Pipe()
        let stderr = Pipe()

        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = [
            "-e",
            "do shell script \"\(command.appleScriptEscaped)\" with administrator privileges",
        ]
        process.standardOutput = stdout
        process.standardError = stderr

        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return .failure(error)
        }

        let output = String(decoding: stdout.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        let errorOutput = String(decoding: stderr.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)

        if process.terminationStatus == 0 {
            return .success(output.trimmingCharacters(in: .whitespacesAndNewlines))
        }

        let message = errorOutput.trimmingCharacters(in: .whitespacesAndNewlines)
        return .failure(ShellCommandError(message: message.isEmpty ? "AppleScript exited with status \(process.terminationStatus)." : message))
    }
}

struct ShellCommandError: LocalizedError {
    let message: String

    var errorDescription: String? {
        message
    }
}

extension String {
    var shellEscaped: String {
        if isEmpty {
            return "''"
        }

        return "'" + replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }

    var appleScriptEscaped: String {
        replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }

    var nonEmpty: String? {
        isEmpty ? nil : self
    }
}
