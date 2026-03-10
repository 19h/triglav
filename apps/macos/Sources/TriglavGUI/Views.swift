import AppKit
import SwiftUI

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - Navigation
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

private enum Panel: String, CaseIterable, Identifiable, Hashable {
    case status, connection, network, activity
    var id: Self { self }

    var icon: String {
        switch self {
        case .status: "gauge.with.dots.needle.33percent"
        case .connection: "slider.horizontal.3"
        case .network: "network"
        case .activity: "terminal"
        }
    }

    var title: String {
        switch self {
        case .status: "Status"
        case .connection: "Connection"
        case .network: "Network"
        case .activity: "Activity"
        }
    }
}

private extension ConnectionState {
    var tint: Color {
        switch self {
        case .disconnected: .secondary
        case .starting: .orange
        case .connected: .green
        case .stopping: .orange
        case .error: .red
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - Demo Data
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#if DEBUG
private enum DemoData {
    static let uplinks: [ClientUplinkSnapshot] = [
        ClientUplinkSnapshot(
            id: "uplink-wifi-0", interface: "en0", remoteAddr: "185.156.46.12:4433",
            state: "active", health: "healthy",
            rttMs: 24.3, lossPercent: 0.2, bandwidthMbps: 87.4,
            bytesSent: 1_284_039_168, bytesReceived: 3_841_507_328,
            natType: "full-cone", externalAddr: "78.90.112.34:52841"
        ),
        ClientUplinkSnapshot(
            id: "uplink-eth-1", interface: "en7", remoteAddr: "185.156.46.12:4433",
            state: "active", health: "healthy",
            rttMs: 11.7, lossPercent: 0.0, bandwidthMbps: 412.8,
            bytesSent: 2_048_576_000, bytesReceived: 6_174_015_488,
            natType: "direct", externalAddr: "203.45.67.89:4433"
        ),
        ClientUplinkSnapshot(
            id: "uplink-cell-2", interface: "pdp_ip0", remoteAddr: "185.156.46.12:4433",
            state: "active", health: "degraded",
            rttMs: 68.1, lossPercent: 2.4, bandwidthMbps: 18.2,
            bytesSent: 156_237_824, bytesReceived: 412_516_352,
            natType: "symmetric", externalAddr: "100.64.0.1:38271"
        ),
    ]

    static let quality = ClientQualitySnapshot(
        usableUplinks: 3, totalUplinks: 3,
        avgRttMs: 34.7, avgLossPercent: 0.87,
        totalBandwidthMbps: 518.4,
        packetsSent: 48_291_043, packetsReceived: 47_872_116, packetsDropped: 418_927
    )

    static let tunnel = ClientTunnelSnapshot(
        tunName: "tg0", fullTunnel: true,
        includeRoutes: ["0.0.0.0/0"], excludeRoutes: ["192.168.1.0/24"]
    )

    static let status = ClientStatusSnapshot(
        version: "0.9.4-dev",
        uptimeSeconds: 14832,
        state: "running",
        role: "client",
        mode: "tun",
        processId: 48291,
        sessionId: "a4c8e1f0-7b32-4d91-b6e2-3f1a9c5d8e72",
        connectionId: "conn-7f3a2b1e",
        quality: quality,
        tunnel: tunnel,
        uplinks: uplinks,
        sessions: [
            ClientSessionSnapshot(
                id: "sess-01", userId: nil,
                remoteAddrs: ["10.0.85.1"],
                connectedAt: "2026-03-09T23:05:12Z",
                bytesSent: 3_488_852_992, bytesReceived: 10_428_039_168,
                uplinksUsed: ["uplink-wifi-0", "uplink-eth-1", "uplink-cell-2"]
            ),
        ],
        totalBytesSent: 3_488_852_992,
        totalBytesReceived: 10_428_039_168,
        totalConnections: 1
    )

    static let interfaces: [NetworkInterface] = [
        NetworkInterface(
            name: "en0", displayName: "Wi-Fi", kind: "Wi-Fi",
            macAddress: "A4:83:E7:2F:1B:C0",
            ipv4Address: "192.168.1.42", ipv4Netmask: "255.255.255.0",
            ipv6Addresses: ["2001:db8::a483:e7ff:fe2f:1bc0"],
            isUp: true, isRunning: true, isLoopback: false,
            isDefaultRoute: true
        ),
        NetworkInterface(
            name: "en7", displayName: "USB 10/100/1000 LAN", kind: "Ethernet",
            macAddress: "00:E0:4C:68:01:A3",
            ipv4Address: "10.0.0.15", ipv4Netmask: "255.255.255.0",
            ipv6Addresses: [],
            isUp: true, isRunning: true, isLoopback: false,
            isDefaultRoute: false
        ),
        NetworkInterface(
            name: "en1", displayName: "Thunderbolt Ethernet Slot 1", kind: "Ethernet",
            macAddress: "82:C5:F2:7A:D4:10",
            ipv4Address: nil, ipv4Netmask: nil,
            ipv6Addresses: [],
            isUp: true, isRunning: false, isLoopback: false,
            isDefaultRoute: false
        ),
        NetworkInterface(
            name: "en2", displayName: "Thunderbolt Bridge", kind: "Bridge",
            macAddress: "82:C5:F2:7A:D4:01",
            ipv4Address: nil, ipv4Netmask: nil,
            ipv6Addresses: [],
            isUp: false, isRunning: false, isLoopback: false,
            isDefaultRoute: false
        ),
        NetworkInterface(
            name: "pdp_ip0", displayName: "iPhone USB", kind: "Cellular",
            macAddress: nil,
            ipv4Address: "172.20.10.3", ipv4Netmask: "255.255.255.240",
            ipv6Addresses: ["2600:1700:6f80:4060::2f"],
            isUp: true, isRunning: true, isLoopback: false,
            isDefaultRoute: false
        ),
    ]

    static let logLines: [String] = [
        "[2026-03-09T23:05:10Z INFO  triglav] Triglav v0.9.4-dev starting",
        "[2026-03-09T23:05:10Z INFO  triglav::config] Loading auth key from argument",
        "[2026-03-09T23:05:10Z INFO  triglav::net] Discovering network interfaces...",
        "[2026-03-09T23:05:10Z INFO  triglav::net] Found en0 (Wi-Fi) — 192.168.1.42",
        "[2026-03-09T23:05:10Z INFO  triglav::net] Found en7 (Ethernet) — 10.0.0.15",
        "[2026-03-09T23:05:10Z INFO  triglav::net] Found pdp_ip0 (Cellular) — 100.64.0.1",
        "[2026-03-09T23:05:11Z INFO  triglav::tun] Creating TUN device tg0",
        "[2026-03-09T23:05:11Z INFO  triglav::tun] Assigning 10.0.85.1/24 to tg0",
        "[2026-03-09T23:05:11Z INFO  triglav::tun] Full tunnel mode: routing 0.0.0.0/0 via tg0",
        "[2026-03-09T23:05:11Z INFO  triglav::tun] Excluding route 192.168.1.0/24",
        "[2026-03-09T23:05:11Z INFO  triglav::conn] Connecting to relay 185.156.46.12:4433",
        "[2026-03-09T23:05:11Z INFO  triglav::conn] Handshake completed (QUIC, 0-RTT)",
        "[2026-03-09T23:05:11Z INFO  triglav::conn] Session a4c8e1f0 established",
        "[2026-03-09T23:05:12Z INFO  triglav::uplink] en0: uplink active (NAT: full-cone, ext 78.90.112.34:52841)",
        "[2026-03-09T23:05:12Z INFO  triglav::uplink] en7: uplink active (NAT: direct, ext 203.45.67.89:4433)",
        "[2026-03-09T23:05:12Z INFO  triglav::uplink] pdp_ip0: uplink active (NAT: symmetric, ext 100.64.0.1:38271)",
        "[2026-03-09T23:05:12Z INFO  triglav] Connected! Tunnel running on tg0 with 3 uplinks",
        "[2026-03-09T23:15:42Z INFO  triglav::sched] Strategy: adaptive — weights en0=0.35, en7=0.55, pdp_ip0=0.10",
        "[2026-03-09T23:28:17Z WARN  triglav::uplink] pdp_ip0: RTT spike 142ms (avg 68ms), reducing weight",
        "[2026-03-09T23:28:18Z INFO  triglav::sched] Rebalanced: en0=0.38, en7=0.57, pdp_ip0=0.05",
        "[2026-03-09T23:45:03Z INFO  triglav::uplink] pdp_ip0: RTT recovered to 65ms, restoring weight",
        "[2026-03-10T00:12:55Z INFO  triglav::stats] TX 3.2 GiB / RX 9.7 GiB — 48.3M packets, 0.87% loss",
        "[2026-03-10T01:30:00Z INFO  triglav::uplink] en7: bandwidth probe complete: 412.8 Mbps",
        "[2026-03-10T02:45:12Z INFO  triglav::uplink] en0: bandwidth probe complete: 87.4 Mbps",
        "[2026-03-10T03:00:01Z INFO  triglav::stats] Uptime 4h 7m — all uplinks healthy",
    ]
}
#endif

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - Main Window
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct MainWindowView: View {
    @EnvironmentObject private var model: AppModel
    @State private var selectedPanel: Panel = .status
    #if DEBUG
    @State private var demoMode = false
    #endif

    var body: some View {
        NavigationSplitView {
            List(Panel.allCases, selection: $selectedPanel) { panel in
                Label(panel.title, systemImage: panel.icon)
                    .tag(panel)
            }
            .listStyle(.sidebar)
            .navigationSplitViewColumnWidth(min: 150, ideal: 170, max: 200)
            #if DEBUG
            .safeAreaInset(edge: .bottom) {
                Toggle("Demo", isOn: $demoMode)
                    .toggleStyle(.switch)
                    .controlSize(.small)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
            }
            #endif
        } detail: {
            detailContent
                .toolbar {
                    ToolbarItemGroup(placement: .primaryAction) {
                        toolbarItems
                    }
                }
        }
    }

    @ViewBuilder
    private var toolbarItems: some View {
        #if DEBUG
        if demoMode {
            Button(action: {}) {
                Label("Disconnect", systemImage: "stop.fill")
            }
        } else {
            connectButton
        }
        #else
        connectButton
        #endif
    }

    private var connectButton: some View {
        Button(action: model.canDisconnect ? model.disconnect : model.connect) {
            Label(
                model.canDisconnect ? "Disconnect" : "Connect",
                systemImage: model.canDisconnect ? "stop.fill" : "play.fill"
            )
        }
        .disabled(!model.canConnect && !model.canDisconnect)
    }

    @ViewBuilder
    private var detailContent: some View {
        #if DEBUG
        if demoMode {
            switch selectedPanel {
            case .status: DemoStatusPanel()
            case .connection: ConnectionPanel()
            case .network: DemoNetworkPanel()
            case .activity: DemoActivityPanel()
            }
        } else {
            realDetailContent
        }
        #else
        realDetailContent
        #endif
    }

    @ViewBuilder
    private var realDetailContent: some View {
        switch selectedPanel {
        case .status: StatusPanel()
        case .connection: ConnectionPanel()
        case .network: NetworkPanel()
        case .activity: ActivityPanel()
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - Status Panel
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

private struct StatusPanel: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            // ── Top: connection state banner ──
            connectionBanner
                .padding(16)

            Divider()

            // ── Bottom: scrollable info table that fills all remaining space ──
            infoTable
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var connectionBanner: some View {
        HStack(spacing: 12) {
            // State indicator
            Circle()
                .fill(model.runtime.state.tint)
                .frame(width: 10, height: 10)
                .overlay(
                    Circle()
                        .fill(model.runtime.state.tint.opacity(0.3))
                        .frame(width: 20, height: 20)
                        .opacity(model.runtime.state == .connected ? 1 : 0)
                )

            VStack(alignment: .leading, spacing: 1) {
                Text(model.runtime.state.title)
                    .font(.headline)
                Text(model.statusHeadline)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            // Quick stats when connected
            if model.runtime.state == .connected, let q = model.liveStatus?.quality {
                HStack(spacing: 16) {
                    quickStat(String(format: "%.0f ms", q.avgRttMs), label: "RTT")
                    quickStat(String(format: "%.1f%%", q.avgLossPercent), label: "Loss")
                    quickStat("\(q.usableUplinks)/\(q.totalUplinks)", label: "Uplinks")
                }
            }
        }
    }

    private func quickStat(_ value: String, label: String) -> some View {
        VStack(alignment: .center, spacing: 1) {
            Text(value)
                .font(.system(size: 13, weight: .medium, design: .rounded))
            Text(label)
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(.tertiary)
        }
    }

    /// A full-height table of all session and system info
    private var infoTable: some View {
        let rows = buildRows()
        return ScrollView {
            LazyVStack(spacing: 0) {
                ForEach(Array(rows.enumerated()), id: \.offset) { idx, row in
                    switch row {
                    case .header(let title):
                        sectionHeader(title)
                    case .row(let label, let value, let tint):
                        infoRow(label: label, value: value, tint: tint, even: idx % 2 == 0)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private enum InfoItem {
        case header(String)
        case row(String, String, Color?)
    }

    private func buildRows() -> [InfoItem] {
        var items: [InfoItem] = []

        items.append(.header("Session"))
        items.append(.row("State", model.runtime.state.title, model.runtime.state.tint))
        items.append(.row("Mode", model.liveStatus?.mode?.uppercased() ?? model.settings.mode.title, nil))
        items.append(.row("Session ID", model.sessionDisplay, nil))
        items.append(.row("Connection ID", model.connectionDisplay, nil))
        items.append(.row("Tunnel", model.tunnelDisplay, nil))

        if model.runtime.state == .connected {
            items.append(.row("Traffic", model.trafficSummary, nil))
            items.append(.row("Quality", model.qualityDisplay, nil))
        }

        if let pid = model.runtime.pid {
            items.append(.row("PID", "\(pid)", nil))
        }

        if let err = model.runtime.lastError {
            items.append(.row("Last Error", err, .red))
        }

        items.append(.header("System"))
        items.append(.row("Endpoint", model.statusEndpointDisplay, nil))
        items.append(.row("Interfaces", model.interfaceSummary, nil))
        items.append(.row("Binary", model.resolvedBinaryDisplay, nil))
        items.append(.row("Logs", model.activeLogLocation, nil))

        // Live uplinks summary when connected
        if !model.liveUplinks.isEmpty {
            items.append(.header("Uplinks"))
            for uplink in model.liveUplinks {
                let detail = [
                    uplink.remoteAddr,
                    "\(uplink.state)/\(uplink.health)",
                    String(format: "RTT %.0fms", uplink.rttMs),
                    String(format: "Loss %.1f%%", uplink.lossPercent),
                    String(format: "%.0f Mbps", uplink.bandwidthMbps),
                ].joined(separator: "  ·  ")
                let color: Color = uplink.health.lowercased() == "healthy" ? .green
                    : uplink.health.lowercased() == "degraded" ? .orange : .red
                items.append(.row(uplink.interface ?? uplink.id, detail, color))
            }
        }

        return items
    }

    private func sectionHeader(_ title: String) -> some View {
        Text(title.uppercased())
            .font(.system(size: 10, weight: .semibold))
            .tracking(0.8)
            .foregroundStyle(.tertiary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 16)
            .padding(.top, 12)
            .padding(.bottom, 4)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
    }

    private func infoRow(label: String, value: String, tint: Color?, even: Bool) -> some View {
        HStack(spacing: 0) {
            HStack(spacing: 6) {
                if let tint {
                    Circle().fill(tint).frame(width: 6, height: 6)
                }
                Text(label)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
            }
            .frame(width: 110, alignment: .trailing)
            .padding(.trailing, 12)

            Text(value)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.primary)
                .textSelection(.enabled)
                .lineLimit(1)
                .truncationMode(.middle)

            Spacer(minLength: 0)
        }
        .padding(.vertical, 5)
        .padding(.horizontal, 16)
        .background(even ? Color.clear : Color(nsColor: .controlBackgroundColor).opacity(0.12))
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - Connection Panel
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

private struct ConnectionPanel: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Form {
            Section("General") {
                Picker("Mode", selection: $model.settings.mode) {
                    ForEach(ConnectionMode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.segmented)

                Picker("Strategy", selection: $model.settings.strategy) {
                    ForEach(SchedulingPreset.allCases) { preset in
                        Text(preset.title).tag(preset)
                    }
                }

                SecureInputRow(
                    label: "Auth Key",
                    text: $model.settings.authKey,
                    placeholder: "tg1_..."
                )
            }

            Section("Interfaces") {
                Toggle("Auto-discover", isOn: $model.settings.autoDiscoverInterfaces)

                if !model.settings.autoDiscoverInterfaces {
                    ForEach(model.availableInterfaces) { iface in
                        Toggle(isOn: Binding(
                            get: { model.settings.selectedInterfaces.contains(iface.name) },
                            set: { _ in model.toggleInterface(iface.name) }
                        )) {
                            HStack(spacing: 6) {
                                Text(iface.displayName)
                                Text(iface.name)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundStyle(.tertiary)
                            }
                        }
                    }
                }
            }

            if model.settings.mode == .tun {
                Section {
                    Toggle("Route all traffic", isOn: $model.settings.fullTunnel)

                    TextField("TUN Device", text: $model.settings.tunName,
                              prompt: Text("tg0"))
                        .font(.system(.body, design: .monospaced))

                    TextField("Tunnel IPv4", text: $model.settings.tunnelIPv4,
                              prompt: Text("10.0.85.1"))
                        .font(.system(.body, design: .monospaced))

                    TextField("Split-tunnel routes", text: $model.settings.routes,
                              prompt: Text("10.0.0.0/8, 172.16.0.0/12"))
                        .font(.system(.body, design: .monospaced))

                    TextField("Excluded routes", text: $model.settings.excludedRoutes,
                              prompt: Text("192.168.0.0/16"))
                        .font(.system(.body, design: .monospaced))
                } header: {
                    Text("Tunnel")
                } footer: {
                    Text("Routes: comma, semicolon, or newline separated.")
                }
            }

            if model.settings.mode == .proxy {
                Section {
                    TextField("SOCKS5 Port", value: $model.settings.socksPort, format: .number)
                        .font(.system(.body, design: .monospaced))
                    TextField("HTTP Proxy Port", value: $model.settings.httpProxyPort, format: .number)
                        .font(.system(.body, design: .monospaced))
                } header: {
                    Text("Proxy")
                } footer: {
                    Text("Set port to 0 to disable.")
                }
            }

            if model.settings.mode.requiresPrivileges {
                Section("Privileged Helper") {
                    LabeledContent("Status") {
                        HelperBadge(state: model.helperStatus.state, title: model.helperStatus.title)
                    }

                    Text(model.helperStatusDetail)
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    HStack(spacing: 8) {
                        if model.helperUsesSMAppService {
                            Button(model.helperNeedsInstall ? "Install" : "Uninstall") {
                                model.helperNeedsInstall ? model.installHelper() : model.uninstallHelper()
                            }
                            .disabled(model.helperActionInProgress)
                        }

                        Button("Refresh", action: model.refreshHelperStatus)
                            .disabled(model.helperActionInProgress)

                        if model.helperUsesSMAppService {
                            Button("Login Items\u{2026}", action: model.openHelperSettings)
                        }

                        if model.helperActionInProgress {
                            ProgressView().controlSize(.small)
                        }
                    }

                    if model.helperStatusDisplayDetail != model.helperStatusDetail {
                        Text(model.helperStatusDisplayDetail)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Section("Command") {
                Text(model.commandPreview)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(6)
                    .background(
                        RoundedRectangle(cornerRadius: 4, style: .continuous)
                            .fill(Color(nsColor: .textBackgroundColor))
                    )

                Button("Copy to Clipboard", action: model.copyCommandToPasteboard)
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.visible)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct SecureInputRow: View {
    let label: String
    @Binding var text: String
    var placeholder: String = ""
    @State private var revealed = false

    var body: some View {
        LabeledContent(label) {
            HStack(spacing: 6) {
                Group {
                    if revealed {
                        TextField(placeholder, text: $text)
                    } else {
                        SecureField(placeholder, text: $text)
                    }
                }
                .font(.system(.body, design: .monospaced))
                .textFieldStyle(.plain)

                Button {
                    revealed.toggle()
                } label: {
                    Image(systemName: revealed ? "eye.slash" : "eye")
                        .font(.system(size: 11))
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
            }
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - Network Panel
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

private struct NetworkPanel: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Group {
            if model.liveUplinks.isEmpty && model.availableInterfaces.isEmpty {
                emptyPlaceholder(icon: "wifi.slash", title: "No Interfaces",
                                 subtitle: "No network interfaces detected.")
            } else {
                networkContent
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .toolbar {
            ToolbarItem(placement: .automatic) {
                Button(action: model.refreshInterfaces) {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
            }
        }
    }

    private var activeInterfaces: [NetworkInterface] {
        model.availableInterfaces.filter(\.isActive)
    }

    private var inactiveInterfaces: [NetworkInterface] {
        model.availableInterfaces.filter { !$0.isActive }
    }

    private var networkContent: some View {
        ScrollView {
            LazyVStack(spacing: 0) {
                // ── Active Uplinks table (when connected) ──
                if !model.liveUplinks.isEmpty {
                    netSectionHeader("Active Uplinks")

                    uplinkColumnHeader

                    ForEach(Array(model.liveUplinks.enumerated()), id: \.element.id) { idx, uplink in
                        uplinkRow(uplink, even: idx % 2 == 0)
                    }
                }

                // ── Active interfaces with full detail ──
                if !activeInterfaces.isEmpty {
                    netSectionHeader("Interfaces")

                    ForEach(activeInterfaces) { iface in
                        InterfaceDetailRow(
                            iface: iface,
                            uplink: model.liveUplinks.first(where: { $0.interface == iface.name })
                        )
                    }
                }

                // ── Inactive interfaces, collapsed ──
                if !inactiveInterfaces.isEmpty {
                    InactiveInterfacesSection(interfaces: inactiveInterfaces)
                }
            }
        }
    }

    private var uplinkColumnHeader: some View {
        HStack(spacing: 0) {
            Text("INTERFACE")
                .frame(maxWidth: .infinity, alignment: .leading)
            Text("STATE")
                .frame(width: 100, alignment: .leading)
            Text("RTT")
                .frame(width: 70, alignment: .trailing)
            Text("LOSS")
                .frame(width: 60, alignment: .trailing)
            Text("BW")
                .frame(width: 80, alignment: .trailing)
        }
        .font(.system(size: 10, weight: .medium))
        .foregroundStyle(.tertiary)
        .padding(.horizontal, 16)
        .padding(.vertical, 5)
        .background(Color(nsColor: .separatorColor).opacity(0.15))
    }

    private func uplinkRow(_ uplink: ClientUplinkSnapshot, even: Bool) -> some View {
        let healthy = uplink.health.lowercased() == "healthy"
        let tint: Color = healthy ? .green : uplink.health.lowercased() == "degraded" ? .orange : .red

        return HStack(spacing: 0) {
            HStack(spacing: 8) {
                Circle().fill(tint).frame(width: 6, height: 6)
                VStack(alignment: .leading, spacing: 1) {
                    Text(uplink.interface ?? uplink.id)
                        .font(.system(size: 12, weight: .medium))
                    Text(uplink.remoteAddr)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Text("\(uplink.state)/\(uplink.health)")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(tint)
                .frame(width: 100, alignment: .leading)

            Text(String(format: "%.0f ms", uplink.rttMs))
                .font(.system(size: 11, design: .monospaced))
                .frame(width: 70, alignment: .trailing)

            Text(String(format: "%.1f%%", uplink.lossPercent))
                .font(.system(size: 11, design: .monospaced))
                .frame(width: 60, alignment: .trailing)

            Text(String(format: "%.0f Mbps", uplink.bandwidthMbps))
                .font(.system(size: 11, design: .monospaced))
                .frame(width: 80, alignment: .trailing)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .background(even ? Color.clear : Color(nsColor: .controlBackgroundColor).opacity(0.12))
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - Activity Panel
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

private struct ActivityPanel: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Group {
            if model.runtime.logLines.isEmpty {
                emptyPlaceholder(icon: "terminal", title: "No Activity",
                                 subtitle: "Connect to begin streaming process output.")
            } else {
                logView
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .toolbar {
            ToolbarItemGroup(placement: .automatic) {
                Button(action: model.openLogsFolder) {
                    Label("Logs", systemImage: "folder")
                }

                Button(action: model.clearLogs) {
                    Label("Clear", systemImage: "trash")
                }
            }
        }
    }

    private var logView: some View {
        ScrollViewReader { proxy in
            ScrollView(.vertical) {
                ScrollView(.horizontal, showsIndicators: false) {
                    VStack(alignment: .leading, spacing: 0) {
                        ForEach(Array(model.runtime.logLines.enumerated()), id: \.offset) { idx, line in
                            let parsed = parseLogLine(line)
                            HStack(alignment: .firstTextBaseline, spacing: 0) {
                                Text("\(idx + 1)")
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundStyle(.quaternary)
                                    .frame(width: 28, alignment: .trailing)
                                    .padding(.trailing, 8)

                                Text(parsed.timestamp)
                                    .font(.system(size: 10.5, design: .monospaced))
                                    .foregroundColor(.gray)
                                    .frame(width: 72, alignment: .leading)
                                    .padding(.trailing, 6)

                                Text(parsed.level)
                                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                                    .foregroundColor(logLevelColor(parsed.level))
                                    .frame(width: 36, alignment: .center)
                                    .padding(.trailing, 6)

                                Text(parsed.module)
                                    .font(.system(size: 10.5, design: .monospaced))
                                    .foregroundColor(.purple)
                                    .frame(width: 130, alignment: .leading)
                                    .padding(.trailing, 6)

                                Text(parsed.message)
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundStyle(logMessageColor(parsed))
                                    .textSelection(.enabled)
                            }
                            .padding(.vertical, 2)
                            .padding(.horizontal, 8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(idx % 2 == 0
                                ? Color.clear
                                : Color(nsColor: .controlBackgroundColor).opacity(0.08))
                            .id(idx)
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
            .background(Color(nsColor: .textBackgroundColor))
            .onChange(of: model.runtime.logLines.count) { _ in
                if let last = model.runtime.logLines.indices.last {
                    proxy.scrollTo(last, anchor: .bottom)
                }
            }
        }
    }

    private struct ParsedLogLine {
        let timestamp: String
        let level: String
        let module: String
        let message: String
    }

    private func parseLogLine(_ line: String) -> ParsedLogLine {
        let stripped = line.trimmingCharacters(in: .whitespaces)

        // GUI system messages
        if stripped.hasPrefix("[gui]") {
            let msg = String(stripped.dropFirst(5)).trimmingCharacters(in: .whitespaces)
            return ParsedLogLine(timestamp: "", level: "GUI", module: "gui", message: msg)
        }

        // Structured: [timestamp LEVEL  module] message
        guard stripped.hasPrefix("["),
              let closeBracket = stripped.firstIndex(of: "]")
        else {
            return ParsedLogLine(timestamp: "", level: "", module: "", message: line)
        }

        let header = stripped[stripped.index(after: stripped.startIndex)..<closeBracket]
        let message = String(stripped[stripped.index(after: closeBracket)...]).trimmingCharacters(in: .whitespaces)

        let parts = header.split(separator: " ", omittingEmptySubsequences: true)
        guard parts.count >= 3 else {
            return ParsedLogLine(timestamp: "", level: "", module: "", message: line)
        }

        let fullTimestamp = String(parts[0])
        let timeOnly: String
        if let tIndex = fullTimestamp.firstIndex(of: "T"),
           let zIndex = fullTimestamp.firstIndex(of: "Z")
        {
            timeOnly = String(fullTimestamp[fullTimestamp.index(after: tIndex)..<zIndex])
        } else {
            timeOnly = fullTimestamp
        }

        let level = String(parts[1])
        let module = String(parts[2...].joined(separator: " "))

        return ParsedLogLine(timestamp: timeOnly, level: level, module: module, message: message)
    }

    private func logLevelColor(_ level: String) -> Color {
        switch level.uppercased() {
        case "ERROR": return .red
        case "WARN": return .orange
        case "INFO": return .green
        case "DEBUG": return .blue
        case "TRACE": return .gray
        case "GUI": return .cyan
        default: return .secondary
        }
    }

    private func logMessageColor(_ parsed: ParsedLogLine) -> Color {
        if parsed.level.uppercased() == "ERROR" { return .red }
        if parsed.level.uppercased() == "WARN" { return .orange }
        if parsed.level.uppercased() == "GUI" { return .cyan }
        let l = parsed.message.lowercased()
        if l.contains("connected!") || l.contains("tunnel running") { return .green }
        return Color.primary.opacity(0.85)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - Demo Panels
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#if DEBUG
private struct DemoStatusPanel: View {
    private let data = DemoData.status
    private let uplinks = DemoData.uplinks

    var body: some View {
        VStack(spacing: 0) {
            connectionBanner
                .padding(16)

            Divider()

            infoTable
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var connectionBanner: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(.green)
                .frame(width: 10, height: 10)
                .overlay(
                    Circle()
                        .fill(Color.green.opacity(0.3))
                        .frame(width: 20, height: 20)
                )

            VStack(alignment: .leading, spacing: 1) {
                Text("Connected")
                    .font(.headline)
                Text("Connected over 3/3 uplinks")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            HStack(spacing: 16) {
                demoQuickStat(String(format: "%.0f ms", data.quality!.avgRttMs), label: "RTT")
                demoQuickStat(String(format: "%.1f%%", data.quality!.avgLossPercent), label: "Loss")
                demoQuickStat("\(data.quality!.usableUplinks)/\(data.quality!.totalUplinks)", label: "Uplinks")
            }
        }
    }

    private func demoQuickStat(_ value: String, label: String) -> some View {
        VStack(alignment: .center, spacing: 1) {
            Text(value)
                .font(.system(size: 13, weight: .medium, design: .rounded))
            Text(label)
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(.tertiary)
        }
    }

    private var infoTable: some View {
        let rows = buildDemoRows()
        return ScrollView {
            LazyVStack(spacing: 0) {
                ForEach(Array(rows.enumerated()), id: \.offset) { idx, row in
                    switch row {
                    case .header(let title):
                        demoSectionHeader(title)
                    case .row(let label, let value, let tint):
                        demoInfoRow(label: label, value: value, tint: tint, even: idx % 2 == 0)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private enum DemoInfoItem {
        case header(String)
        case row(String, String, Color?)
    }

    private func buildDemoRows() -> [DemoInfoItem] {
        var items: [DemoInfoItem] = []

        items.append(.header("Session"))
        items.append(.row("State", "Connected", .green))
        items.append(.row("Mode", "TUN", nil))
        items.append(.row("Session ID", "a4c8e1f0", nil))
        items.append(.row("Connection ID", "conn-7f3a2b1e", nil))
        items.append(.row("Tunnel", "tg0 \u{00B7} full tunnel", nil))
        items.append(.row("Traffic", "TX 3.2 GiB \u{00B7} RX 9.7 GiB", nil))
        items.append(.row("Quality", "34.7 ms RTT \u{00B7} 0.9% loss \u{00B7} 518.4 Mbps aggregate", nil))
        items.append(.row("PID", "48291", nil))
        items.append(.row("Uptime", "4h 7m 12s", nil))

        items.append(.header("System"))
        items.append(.row("Endpoint", "http://127.0.0.1:9091/status", nil))
        items.append(.row("Interfaces", "en0, en7, pdp_ip0", nil))
        items.append(.row("Binary", "/opt/homebrew/bin/triglav", nil))
        items.append(.row("Logs", "~/Library/Logs/TriglavGUI/tun-20260309-230510.log", nil))
        items.append(.row("Version", "0.9.4-dev", nil))

        items.append(.header("Uplinks"))
        for uplink in uplinks {
            let detail = [
                uplink.remoteAddr,
                "\(uplink.state)/\(uplink.health)",
                String(format: "RTT %.0fms", uplink.rttMs),
                String(format: "Loss %.1f%%", uplink.lossPercent),
                String(format: "%.0f Mbps", uplink.bandwidthMbps),
            ].joined(separator: "  \u{00B7}  ")
            let color: Color = uplink.health.lowercased() == "healthy" ? .green
                : uplink.health.lowercased() == "degraded" ? .orange : .red
            items.append(.row(uplink.interface ?? uplink.id, detail, color))
        }

        items.append(.header("Routing"))
        items.append(.row("Include", "0.0.0.0/0 (full tunnel)", nil))
        items.append(.row("Exclude", "192.168.1.0/24", nil))

        items.append(.header("Packets"))
        items.append(.row("Sent", "48,291,043", nil))
        items.append(.row("Received", "47,872,116", nil))
        items.append(.row("Dropped", "418,927 (0.87%)", nil))

        return items
    }
}

private struct DemoNetworkPanel: View {
    private let uplinks = DemoData.uplinks
    private let interfaces = DemoData.interfaces

    private var activeInterfaces: [NetworkInterface] {
        interfaces.filter(\.isActive)
    }

    private var inactiveInterfaces: [NetworkInterface] {
        interfaces.filter { !$0.isActive }
    }

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 0) {
                // ── Active Uplinks (when connected) ──
                netSectionHeader("Active Uplinks")

                uplinkColumnHeader

                ForEach(Array(uplinks.enumerated()), id: \.element.id) { idx, uplink in
                    uplinkRow(uplink, even: idx % 2 == 0)
                }

                // ── Active interfaces with full detail ──
                netSectionHeader("Interfaces")

                ForEach(activeInterfaces) { iface in
                    InterfaceDetailRow(iface: iface, uplink: uplinks.first(where: { $0.interface == iface.name }))
                }

                // ── Inactive interfaces, collapsed ──
                if !inactiveInterfaces.isEmpty {
                    InactiveInterfacesSection(interfaces: inactiveInterfaces)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var uplinkColumnHeader: some View {
        HStack(spacing: 0) {
            Text("INTERFACE")
                .frame(maxWidth: .infinity, alignment: .leading)
            Text("STATE")
                .frame(width: 100, alignment: .leading)
            Text("RTT")
                .frame(width: 70, alignment: .trailing)
            Text("LOSS")
                .frame(width: 60, alignment: .trailing)
            Text("BW")
                .frame(width: 80, alignment: .trailing)
        }
        .font(.system(size: 10, weight: .medium))
        .foregroundStyle(.tertiary)
        .padding(.horizontal, 16)
        .padding(.vertical, 5)
        .background(Color(nsColor: .separatorColor).opacity(0.15))
    }

    private func uplinkRow(_ uplink: ClientUplinkSnapshot, even: Bool) -> some View {
        let healthy = uplink.health.lowercased() == "healthy"
        let tint: Color = healthy ? .green : uplink.health.lowercased() == "degraded" ? .orange : .red

        return HStack(spacing: 0) {
            HStack(spacing: 8) {
                Circle().fill(tint).frame(width: 6, height: 6)
                VStack(alignment: .leading, spacing: 1) {
                    Text(uplink.interface ?? uplink.id)
                        .font(.system(size: 12, weight: .medium))
                    Text(uplink.remoteAddr)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Text("\(uplink.state)/\(uplink.health)")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(tint)
                .frame(width: 100, alignment: .leading)

            Text(String(format: "%.0f ms", uplink.rttMs))
                .font(.system(size: 11, design: .monospaced))
                .frame(width: 70, alignment: .trailing)

            Text(String(format: "%.1f%%", uplink.lossPercent))
                .font(.system(size: 11, design: .monospaced))
                .frame(width: 60, alignment: .trailing)

            Text(String(format: "%.0f Mbps", uplink.bandwidthMbps))
                .font(.system(size: 11, design: .monospaced))
                .frame(width: 80, alignment: .trailing)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .background(even ? Color.clear : Color(nsColor: .controlBackgroundColor).opacity(0.12))
    }
}

private struct DemoActivityPanel: View {
    private let logLines = DemoData.logLines

    private struct ParsedLog {
        let timestamp: String
        let level: String
        let module: String
        let message: String
    }

    var body: some View {
        ScrollView(.vertical) {
            ScrollView(.horizontal, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(logLines.enumerated()), id: \.offset) { idx, line in
                        let parsed = parseLine(line)
                        HStack(alignment: .firstTextBaseline, spacing: 0) {
                            // Line number
                            Text("\(idx + 1)")
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(.quaternary)
                                .frame(width: 28, alignment: .trailing)
                                .padding(.trailing, 8)

                            // Timestamp
                            Text(parsed.timestamp)
                                .font(.system(size: 10.5, design: .monospaced))
                                .foregroundColor(.gray)
                                .frame(width: 72, alignment: .leading)
                                .padding(.trailing, 6)

                            // Level badge
                            Text(parsed.level)
                                .font(.system(size: 9, weight: .bold, design: .monospaced))
                                .foregroundColor(levelColor(parsed.level))
                                .frame(width: 36, alignment: .center)
                                .padding(.trailing, 6)

                            // Module
                            Text(parsed.module)
                                .font(.system(size: 10.5, design: .monospaced))
                                .foregroundColor(.purple)
                                .frame(width: 130, alignment: .leading)
                                .padding(.trailing, 6)

                            // Message
                            Text(parsed.message)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(messageColor(parsed))
                                .textSelection(.enabled)
                        }
                        .padding(.vertical, 2)
                        .padding(.horizontal, 8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(idx % 2 == 0
                            ? Color.clear
                            : Color(nsColor: .controlBackgroundColor).opacity(0.08))
                    }
                }
                .padding(.vertical, 4)
            }
        }
        .background(Color(nsColor: .textBackgroundColor))
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// Parse "[2026-03-09T23:05:10Z INFO  triglav::net] Found en0 ..."
    private func parseLine(_ line: String) -> ParsedLog {
        // Try to parse structured log format: [timestamp LEVEL  module] message
        let stripped = line.trimmingCharacters(in: .whitespaces)
        guard stripped.hasPrefix("["),
              let closeBracket = stripped.firstIndex(of: "]")
        else {
            return ParsedLog(timestamp: "", level: "INFO", module: "", message: line)
        }

        let header = stripped[stripped.index(after: stripped.startIndex)..<closeBracket]
        let message = String(stripped[stripped.index(after: closeBracket)...]).trimmingCharacters(in: .whitespaces)

        let parts = header.split(separator: " ", omittingEmptySubsequences: true)
        guard parts.count >= 3 else {
            return ParsedLog(timestamp: "", level: "INFO", module: "", message: line)
        }

        // Timestamp: "2026-03-09T23:05:10Z" -> show just time "23:05:10"
        let fullTimestamp = String(parts[0])
        let timeOnly: String
        if let tIndex = fullTimestamp.firstIndex(of: "T"),
           let zIndex = fullTimestamp.firstIndex(of: "Z")
        {
            timeOnly = String(fullTimestamp[fullTimestamp.index(after: tIndex)..<zIndex])
        } else {
            timeOnly = fullTimestamp
        }

        let level = String(parts[1])
        let module = String(parts[2...].joined(separator: " "))

        return ParsedLog(timestamp: timeOnly, level: level, module: module, message: message)
    }

    private func levelColor(_ level: String) -> Color {
        switch level.uppercased() {
        case "ERROR": return .red
        case "WARN": return .orange
        case "INFO": return .green
        case "DEBUG": return .blue
        case "TRACE": return .gray
        default: return .secondary
        }
    }

    private func messageColor(_ parsed: ParsedLog) -> Color {
        let l = parsed.message.lowercased()
        if parsed.level == "ERROR" { return .red }
        if parsed.level == "WARN" { return .orange }
        if l.contains("connected!") || l.contains("tunnel running") { return .green }
        return Color.primary.opacity(0.85)
    }
}

// Shared helpers used by both real and demo panels
private func demoSectionHeader(_ title: String) -> some View {
    Text(title.uppercased())
        .font(.system(size: 10, weight: .semibold))
        .tracking(0.8)
        .foregroundStyle(.tertiary)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.top, 12)
        .padding(.bottom, 4)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.3))
}

private func demoInfoRow(label: String, value: String, tint: Color?, even: Bool) -> some View {
    HStack(spacing: 0) {
        HStack(spacing: 6) {
            if let tint {
                Circle().fill(tint).frame(width: 6, height: 6)
            }
            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .frame(width: 110, alignment: .trailing)
        .padding(.trailing, 12)

        Text(value)
            .font(.system(size: 11, design: .monospaced))
            .foregroundStyle(.primary)
            .textSelection(.enabled)
            .lineLimit(1)
            .truncationMode(.middle)

        Spacer(minLength: 0)
    }
    .padding(.vertical, 5)
    .padding(.horizontal, 16)
    .background(even ? Color.clear : Color(nsColor: .controlBackgroundColor).opacity(0.12))
}
#endif

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - Shared Components
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

private func netSectionHeader(_ title: String) -> some View {
    Text(title.uppercased())
        .font(.system(size: 10, weight: .semibold))
        .tracking(0.8)
        .foregroundStyle(.tertiary)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.top, 14)
        .padding(.bottom, 4)
}

/// Whether an interface is "active" — has an IP and link is up
private extension NetworkInterface {
    var isActive: Bool {
        isUp && isRunning && (ipv4Address != nil || !ipv6Addresses.isEmpty)
    }
}

/// Full detail row for an active network interface
private struct InterfaceDetailRow: View {
    let iface: NetworkInterface
    let uplink: ClientUplinkSnapshot?

    var body: some View {
        VStack(spacing: 0) {
            Divider().padding(.leading, 16)

            VStack(alignment: .leading, spacing: 8) {
                // ── Header: icon, name, status badge ──
                HStack(spacing: 10) {
                    Image(systemName: ifaceIcon(iface.kind))
                        .font(.system(size: 14))
                        .foregroundColor(statusColor)
                        .frame(width: 20)

                    VStack(alignment: .leading, spacing: 1) {
                        HStack(spacing: 6) {
                            Text(iface.displayName)
                                .font(.system(size: 12, weight: .semibold))

                            if iface.isDefaultRoute {
                                Text("DEFAULT")
                                    .font(.system(size: 8, weight: .bold))
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 5)
                                    .padding(.vertical, 1)
                                    .background(Color.accentColor, in: Capsule())
                            }
                        }

                        HStack(spacing: 6) {
                            Text(iface.name)
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(.tertiary)

                            Text("\u{00B7}")
                                .foregroundStyle(.quaternary)

                            Text(iface.kind)
                                .font(.system(size: 10))
                                .foregroundStyle(.tertiary)

                            Text("\u{00B7}")
                                .foregroundStyle(.quaternary)

                            Text(iface.statusDescription)
                                .font(.system(size: 10, weight: .medium))
                                .foregroundColor(statusColor)
                        }
                    }

                    Spacer()

                    Circle()
                        .fill(statusColor)
                        .frame(width: 8, height: 8)
                }

                // ── Detail grid ──
                detailGrid
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
        }
    }

    private var detailGrid: some View {
        let rows = buildDetailRows()
        return VStack(spacing: 3) {
            ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                HStack(spacing: 0) {
                    Text(row.label)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                        .frame(width: 80, alignment: .trailing)
                        .padding(.trailing, 10)

                    Text(row.value)
                        .font(.system(size: 10.5, design: .monospaced))
                        .foregroundStyle(.primary)
                        .textSelection(.enabled)
                        .lineLimit(1)

                    Spacer(minLength: 0)
                }
            }
        }
        .padding(.leading, 30)
    }

    private struct DetailRow {
        let label: String
        let value: String
    }

    private func buildDetailRows() -> [DetailRow] {
        var rows: [DetailRow] = []

        if let ipv4 = iface.ipv4CIDR {
            rows.append(DetailRow(label: "IPv4", value: ipv4))
        } else if iface.ipv4Address != nil {
            rows.append(DetailRow(label: "IPv4", value: iface.ipv4Address!))
        }

        for ipv6 in iface.ipv6Addresses {
            rows.append(DetailRow(label: rows.contains(where: { $0.label == "IPv6" }) ? "" : "IPv6", value: ipv6))
        }

        if let mac = iface.macAddress {
            rows.append(DetailRow(label: "MAC", value: mac))
        }

        if let uplink = uplink {
            rows.append(DetailRow(label: "Remote", value: uplink.remoteAddr))
            rows.append(DetailRow(label: "NAT", value: uplink.natType + (uplink.externalAddr.map { "  \($0)" } ?? "")))
            rows.append(DetailRow(label: "Traffic", value: "TX \(uplink.bytesSent.byteCountDisplay)  RX \(uplink.bytesReceived.byteCountDisplay)"))
        }

        return rows
    }

    private var statusColor: Color {
        if !iface.isUp { return .secondary }
        if !iface.isRunning { return .orange }
        if iface.ipv4Address == nil && iface.ipv6Addresses.isEmpty { return .yellow }
        return .green
    }

    private func ifaceIcon(_ kind: String) -> String {
        switch kind.lowercased() {
        case "wi-fi": return "wifi"
        case "ethernet": return "cable.connector.horizontal"
        case "cellular": return "antenna.radiowaves.left.and.right"
        case "bluetooth": return "wave.3.right"
        case "bridge": return "square.on.square.intersection.dashed"
        case "vpn": return "lock.shield"
        case "loopback": return "arrow.triangle.2.circlepath"
        case "triglav": return "bolt.horizontal"
        default: return "network"
        }
    }
}

/// Compact single-line row for an inactive/down interface
private struct InactiveInterfaceRow: View {
    let iface: NetworkInterface

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: ifaceIcon(iface.kind))
                .font(.system(size: 11))
                .foregroundColor(.secondary)
                .frame(width: 18)

            Text(iface.displayName)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)

            Text(iface.name)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.quaternary)

            Spacer()

            Text(iface.statusDescription)
                .font(.system(size: 10))
                .foregroundStyle(.quaternary)

            if let mac = iface.macAddress {
                Text(mac)
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.quaternary)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 4)
    }

    private func ifaceIcon(_ kind: String) -> String {
        switch kind.lowercased() {
        case "wi-fi": return "wifi"
        case "ethernet": return "cable.connector.horizontal"
        case "cellular": return "antenna.radiowaves.left.and.right"
        case "bluetooth": return "wave.3.right"
        case "bridge": return "square.on.square.intersection.dashed"
        case "vpn": return "lock.shield"
        case "loopback": return "arrow.triangle.2.circlepath"
        default: return "network"
        }
    }
}

/// Collapsible section showing inactive interfaces in compact form
private struct InactiveInterfacesSection: View {
    let interfaces: [NetworkInterface]
    @State private var expanded = false

    var body: some View {
        VStack(spacing: 0) {
            // ── Toggle header ──
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    expanded.toggle()
                }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: expanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.tertiary)
                        .frame(width: 12)

                    Text("INACTIVE (\(interfaces.count))")
                        .font(.system(size: 10, weight: .semibold))
                        .tracking(0.8)
                        .foregroundStyle(.tertiary)

                    Spacer()
                }
                .padding(.horizontal, 16)
                .padding(.top, 14)
                .padding(.bottom, 6)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            // ── Collapsed rows ──
            if expanded {
                ForEach(interfaces) { iface in
                    InactiveInterfaceRow(iface: iface)
                }
            }
        }
    }
}

private func emptyPlaceholder(icon: String, title: String, subtitle: String) -> some View {
    VStack(spacing: 10) {
        Image(systemName: icon)
            .font(.system(size: 32, weight: .ultraLight))
            .foregroundStyle(.quaternary)
        Text(title)
            .font(.headline)
            .foregroundStyle(.secondary)
        Text(subtitle)
            .font(.subheadline)
            .foregroundStyle(.tertiary)
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
}

private struct HelperBadge: View {
    let state: HelperServiceSnapshot.State
    let title: String

    private var color: Color {
        switch state {
        case .ready: .green
        case .approvalRequired: .orange
        case .notRegistered, .fallback: .secondary
        case .notFound, .unknown: .red
        }
    }

    var body: some View {
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(title)
                .font(.system(size: 11, weight: .medium))
        }
        .foregroundStyle(color)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - Menu Bar
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct MenuBarContentView: View {
    @Environment(\.openWindow) private var openWindow
    @ObservedObject var state: MenuBarState
    let connectAction: () -> Void
    let disconnectAction: () -> Void
    let quitAction: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Circle().fill(state.connectionState.tint).frame(width: 8, height: 8)
                VStack(alignment: .leading, spacing: 0) {
                    Text(state.connectionState.title)
                        .font(.system(size: 13, weight: .semibold))
                    Text(state.modeTitle)
                        .font(.system(size: 10)).foregroundStyle(.secondary)
                }
                Spacer()
            }

            Divider()

            Button(state.canDisconnect ? "Disconnect" : "Connect") {
                state.canDisconnect ? disconnectAction() : connectAction()
            }
            .font(.system(size: 12, weight: .medium))

            Button("Open Window") {
                openWindow(id: "main")
                NSApplication.shared.activate(ignoringOtherApps: true)
            }
            .font(.system(size: 12, weight: .medium))

            Button("Settings\u{2026}") {
                NSApplication.shared.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
                NSApplication.shared.activate(ignoringOtherApps: true)
            }
            .font(.system(size: 12))
            .foregroundStyle(.secondary)

            Divider()

            Button("Quit Triglav") { quitAction() }
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
        .padding(12)
        .frame(width: 200)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MARK: - Settings
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct SettingsView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Form {
            Section("CLI Integration") {
                TextField("Binary Path", text: $model.settings.binaryPath,
                          prompt: Text("Auto-detect triglav"))
                    .textFieldStyle(.roundedBorder)
                Text("Leave blank to auto-detect from PATH or Cargo build outputs.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button("Open Log Folder", action: model.openLogsFolder)
            }

            Section("Presence") {
                Toggle("Show status bar icon", isOn: $model.settings.showStatusBarIcon)
                Text("Hiding the icon keeps the app in the Dock.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Privileged Helper") {
                LabeledContent("Status") {
                    HelperBadge(state: model.helperStatus.state, title: model.helperStatus.title)
                }
                Text(model.helperStatusDetail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                HStack {
                    if model.helperUsesSMAppService {
                        Button(model.helperNeedsInstall ? "Install" : "Uninstall") {
                            model.helperNeedsInstall ? model.installHelper() : model.uninstallHelper()
                        }
                        .disabled(model.helperActionInProgress)
                    }
                    Button("Refresh", action: model.refreshHelperStatus)
                        .disabled(model.helperActionInProgress)
                    if model.helperUsesSMAppService {
                        Button("Login Items\u{2026}", action: model.openHelperSettings)
                    }
                }
                if model.helperActionInProgress {
                    ProgressView().controlSize(.small)
                }
                if model.helperStatusDisplayDetail != model.helperStatusDetail {
                    Text(model.helperStatusDisplayDetail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Section("Resolved Paths") {
                LabeledContent("Binary") {
                    Text(model.resolvedBinaryDisplay)
                        .font(.caption)
                        .textSelection(.enabled)
                }
                LabeledContent("Endpoint") {
                    Text(model.statusEndpointDisplay)
                        .font(.caption)
                        .textSelection(.enabled)
                }
                LabeledContent("Logs") {
                    Text(model.activeLogLocation)
                        .font(.caption)
                        .textSelection(.enabled)
                }
            }
        }
        .formStyle(.grouped)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
