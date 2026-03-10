import AppKit
import SwiftUI

private enum LayoutMetrics {
    static let contentWidth: CGFloat = 1118
}

private enum MainWindowPanel: String, CaseIterable, Identifiable {
    case status
    case connection
    case network
    case activity

    var id: Self { self }

    var title: String {
        switch self {
        case .status:
            "Status"
        case .connection:
            "Connection"
        case .network:
            "Network"
        case .activity:
            "Activity"
        }
    }
}

private extension ConnectionState {
    var tint: Color {
        switch self {
        case .disconnected:
            Color(red: 0.55, green: 0.62, blue: 0.72)
        case .starting:
            Color(red: 0.93, green: 0.68, blue: 0.30)
        case .connected:
            Color(red: 0.29, green: 0.79, blue: 0.70)
        case .stopping:
            Color(red: 0.96, green: 0.56, blue: 0.36)
        case .error:
            Color(red: 0.94, green: 0.39, blue: 0.39)
        }
    }
}

struct MainWindowView: View {
    @EnvironmentObject private var model: AppModel
    @State private var selectedPanel: MainWindowPanel = .status

    var body: some View {
        ZStack {
            background

            VStack(alignment: .leading, spacing: 22) {
                heroCard
                    .frame(width: LayoutMetrics.contentWidth, alignment: .leading)

                panelPicker
                    .frame(width: LayoutMetrics.contentWidth, alignment: .leading)

                activePanel
                    .frame(width: LayoutMetrics.contentWidth, alignment: .topLeading)

                Spacer(minLength: 0)
            }
            .padding(28)
            .frame(width: LayoutMetrics.contentWidth)
            .frame(maxHeight: .infinity, alignment: .topLeading)
        }
    }

    private var panelPicker: some View {
        HStack(spacing: 10) {
            ForEach(MainWindowPanel.allCases) { panel in
                Button {
                    selectedPanel = panel
                } label: {
                    Text(panel.title)
                        .font(.system(size: 12, weight: .bold, design: .rounded))
                        .frame(width: 120)
                        .padding(.vertical, 9)
                }
                .buttonStyle(.plain)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(selectedPanel == panel ? Color.white.opacity(0.18) : Color.white.opacity(0.06))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(selectedPanel == panel ? Color.white.opacity(0.18) : Color.white.opacity(0.08), lineWidth: 1)
                )
                .foregroundStyle(selectedPanel == panel ? Color.white : Color.secondary)
            }
        }
    }

    @ViewBuilder
    private var activePanel: some View {
        switch selectedPanel {
        case .status:
            scrollingPanel {
                overviewCard
            }
        case .connection:
            scrollingPanel {
                connectionCard
            }
        case .network:
            interfacesCard
        case .activity:
            logCard
        }
    }

    private func scrollingPanel<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        ScrollView(.vertical) {
            content()
                .frame(width: LayoutMetrics.contentWidth, alignment: .topLeading)
                .padding(.trailing, 10)
        }
        .scrollIndicators(.visible)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private var background: some View {
        LinearGradient(
            colors: [
                Color(red: 0.06, green: 0.10, blue: 0.15),
                Color(red: 0.10, green: 0.14, blue: 0.18),
                Color(red: 0.17, green: 0.13, blue: 0.11),
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        .ignoresSafeArea()
    }

    private var heroCard: some View {
        DashboardCard {
            VStack(alignment: .leading, spacing: 14) {
                StateBadge(state: model.runtime.state)

                Text("Triglav for macOS")
                    .font(.system(size: 32, weight: .bold, design: .rounded))

                Text("A native control room for the multipath client with menu bar controls, TUN orchestration, and live uplink telemetry.")
                    .font(.system(size: 14, weight: .medium, design: .rounded))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                Text(model.statusHeadline)
                    .font(.system(size: 15, weight: .semibold, design: .rounded))
                    .foregroundStyle(.primary)

                Text(model.statusDetail)
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)

                HStack(spacing: 12) {
                    Button(action: model.canDisconnect ? model.disconnect : model.connect) {
                        Label(
                            model.canDisconnect ? "Disconnect" : "Connect",
                            systemImage: model.canDisconnect ? "xmark.circle.fill" : "play.circle.fill"
                        )
                        .frame(minWidth: 132)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)

                    Button("Copy Command", action: model.copyCommandToPasteboard)
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                }
            }
        }
    }

    private var connectionCard: some View {
        DashboardCard(title: "Connection") {
            VStack(alignment: .leading, spacing: 18) {
                HStack(spacing: 16) {
                    LabeledPicker(
                        title: "Mode",
                        selection: $model.settings.mode,
                        values: ConnectionMode.allCases,
                        titleKeyPath: \.title
                    )

                    LabeledPicker(
                        title: "Strategy",
                        selection: $model.settings.strategy,
                        values: SchedulingPreset.allCases,
                        titleKeyPath: \.title
                    )
                }

                VStack(alignment: .leading, spacing: 8) {
                    SectionLabel("Auth key")
                    TextField("tg1_...", text: $model.settings.authKey)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                        .frame(maxWidth: 440, alignment: .leading)
                }

                if model.settings.mode.requiresPrivileges {
                    helperStatusSection
                }

                Toggle("Auto-discover interfaces", isOn: $model.settings.autoDiscoverInterfaces)
                    .toggleStyle(.switch)

                if !model.settings.autoDiscoverInterfaces {
                    VStack(alignment: .leading, spacing: 10) {
                        SectionLabel("Pinned interfaces")

                        List(model.availableInterfaces) { interface in
                            Toggle(isOn: interfaceSelectionBinding(for: interface.name)) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(interface.displayName)
                                        .font(.system(size: 12, weight: .semibold, design: .rounded))
                                    Text("\(interface.name) \u{00B7} \(interface.kind)")
                                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                        .listStyle(.plain)
                        .frame(height: 180)
                    }
                }

                if model.settings.mode == .tun {
                    tunFields
                } else {
                    proxyFields
                }

                VStack(alignment: .leading, spacing: 8) {
                    SectionLabel("Command preview")
                    Text(model.commandPreview)
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private var helperStatusSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                SectionLabel("Privileged helper")
                HelperStateBadge(snapshot: model.helperStatus)
            }

            Text(model.helperStatusDetail)
                .font(.system(size: 12, weight: .medium, design: .rounded))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 12) {
                if model.helperUsesSMAppService {
                    Button(model.helperNeedsInstall ? "Install Helper" : "Uninstall Helper") {
                        if model.helperNeedsInstall {
                            model.installHelper()
                        } else {
                            model.uninstallHelper()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.helperActionInProgress)
                }

                Button("Refresh helper status", action: model.refreshHelperStatus)
                    .buttonStyle(.bordered)
                    .disabled(model.helperActionInProgress)

                if model.helperUsesSMAppService {
                    Button("Open Login Items", action: model.openHelperSettings)
                        .buttonStyle(.bordered)
                }
            }

            if model.helperActionInProgress {
                ProgressView()
                    .controlSize(.small)
            }

            if model.helperStatusDisplayDetail != model.helperStatusDetail {
                Text(model.helperStatusDisplayDetail)
                    .font(.system(size: 11, weight: .medium, design: .rounded))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var tunFields: some View {
        VStack(alignment: .leading, spacing: 14) {
            Toggle("Route all traffic through the tunnel", isOn: $model.settings.fullTunnel)
                .toggleStyle(.switch)

            HStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 8) {
                    SectionLabel("TUN device")
                    TextField("tg0", text: $model.settings.tunName)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 180, alignment: .leading)
                }

                VStack(alignment: .leading, spacing: 8) {
                    SectionLabel("Tunnel IPv4")
                    TextField("10.0.85.1", text: $model.settings.tunnelIPv4)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                        .frame(width: 180, alignment: .leading)
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                SectionLabel("Split-tunnel routes")
                NotesEditor(text: $model.settings.routes, prompt: "10.0.0.0/8, 172.16.0.0/12")
            }

            VStack(alignment: .leading, spacing: 8) {
                SectionLabel("Excluded routes")
                NotesEditor(text: $model.settings.excludedRoutes, prompt: "192.168.0.0/16")
            }
        }
    }

    private var proxyFields: some View {
        HStack(spacing: 16) {
            PortField(title: "SOCKS5 port", value: $model.settings.socksPort)
            PortField(title: "HTTP proxy port", value: $model.settings.httpProxyPort)
        }
    }

    private var overviewCard: some View {
        DashboardCard(title: "Live Status") {
            VStack(alignment: .leading, spacing: 14) {
                KeyValueRow(label: "State", value: model.runtime.state.title)
                KeyValueRow(label: "Mode", value: model.liveStatus?.mode?.uppercased() ?? model.settings.mode.title)
                KeyValueRow(label: "Session", value: model.sessionDisplay)
                KeyValueRow(label: "Connection", value: model.connectionDisplay)
                KeyValueRow(label: "Traffic", value: model.trafficSummary)
                KeyValueRow(label: "Quality", value: model.qualityDisplay)
                KeyValueRow(label: "Tunnel", value: model.tunnelDisplay)
                KeyValueRow(label: "Endpoint", value: model.statusEndpointDisplay)
                KeyValueRow(label: "Interfaces", value: model.interfaceSummary)
                KeyValueRow(label: "Binary", value: model.resolvedBinaryDisplay)
                KeyValueRow(label: "Logs", value: model.activeLogLocation)
            }
        }
    }

    private var interfacesCard: some View {
        DashboardCard(title: model.liveUplinks.isEmpty ? "Detected Interfaces" : "Live Uplinks") {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text(model.liveUplinks.isEmpty ? "macOS network services" : "runtime uplink telemetry")
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundStyle(.secondary)

                    Spacer()

                    Button("Refresh", action: model.refreshInterfaces)
                        .buttonStyle(.borderless)
                }

                if !model.liveUplinks.isEmpty {
                    List(model.liveUplinks) { uplink in
                        LiveUplinkRow(
                            uplink: uplink,
                            tint: color(for: uplink.state, health: uplink.health)
                        )
                    }
                    .listStyle(.plain)
                    .frame(minHeight: 320)
                } else if model.availableInterfaces.isEmpty {
                    Text("No interfaces were discovered through System Configuration.")
                        .font(.system(size: 12, weight: .medium, design: .rounded))
                        .foregroundStyle(.secondary)
                } else {
                    List(model.availableInterfaces) { interface in
                        InterfaceStatusRow(
                            interface: interface,
                            selected: model.settings.selectedInterfaces.contains(interface.name)
                        )
                    }
                    .listStyle(.plain)
                    .frame(minHeight: 320)
                }
            }
        }
    }

    private var logCard: some View {
        DashboardCard(title: "Activity") {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Recent output")
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                        .foregroundStyle(.secondary)

                    Spacer()

                    Button("Open Logs", action: model.openLogsFolder)
                        .buttonStyle(.borderless)

                    Button("Clear", action: model.clearLogs)
                        .buttonStyle(.borderless)
                }

                if model.runtime.logLines.isEmpty {
                    Text("No output yet. Connect to stream Triglav logs here.")
                        .font(.system(size: 12, weight: .medium, design: .rounded))
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.vertical, 6)
                        .frame(minHeight: 220, alignment: .topLeading)
                        .padding(14)
                        .background(
                            RoundedRectangle(cornerRadius: 18, style: .continuous)
                                .fill(Color.black.opacity(0.20))
                        )
                } else {
                    List(Array(model.runtime.logLines.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                            .foregroundStyle(.primary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .listStyle(.plain)
                    .frame(minHeight: 260)
                }
            }
        }
    }

    private func interfaceSelectionBinding(for interfaceName: String) -> Binding<Bool> {
        Binding(
            get: { model.settings.selectedInterfaces.contains(interfaceName) },
            set: { isSelected in
                let alreadySelected = model.settings.selectedInterfaces.contains(interfaceName)
                guard isSelected != alreadySelected else { return }
                model.toggleInterface(interfaceName)
            }
        )
    }

    private func color(for state: String, health: String) -> Color {
        switch (state.lowercased(), health.lowercased()) {
        case ("connected", "healthy"), ("connected", "degraded"):
            ConnectionState.connected.tint
        case ("connecting", _), ("handshaking", _):
            ConnectionState.starting.tint
        default:
            ConnectionState.error.tint
        }
    }
}

struct MenuBarContentView: View {
    @Environment(\.openWindow) private var openWindow
    @ObservedObject var state: MenuBarState
    let connectAction: () -> Void
    let disconnectAction: () -> Void
    let quitAction: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            StateBadge(state: state.connectionState)

            Text(state.connectionState.title)
                .font(.system(size: 14, weight: .bold, design: .rounded))

            Text(state.modeTitle)
                .font(.system(size: 11, weight: .medium, design: .rounded))
                .foregroundStyle(.secondary)

            Divider()

            Button(state.canDisconnect ? "Disconnect" : "Connect") {
                if state.canDisconnect {
                    disconnectAction()
                } else {
                    connectAction()
                }
            }

            Button("Open Window") {
                openWindow(id: "main")
                NSApplication.shared.activate(ignoringOtherApps: true)
            }

            Button("Settings") {
                NSApplication.shared.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
                NSApplication.shared.activate(ignoringOtherApps: true)
            }

            Divider()

            Button("Quit Triglav") {
                quitAction()
            }
        }
        .padding(14)
        .frame(width: 240)
    }
}

struct SettingsView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Form {
            Section("CLI Integration") {
                TextField("Auto-detect triglav", text: $model.settings.binaryPath)
                    .textFieldStyle(.roundedBorder)

                Text("Leave the field blank to auto-detect `triglav` from PATH or nearby Cargo build outputs.")
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .foregroundStyle(.secondary)

                Button("Open log folder", action: model.openLogsFolder)
            }

            Section("Presence") {
                Toggle("Show status bar icon", isOn: $model.settings.showStatusBarIcon)

                Text("If you hide the status bar icon, the app stays in the Dock so the main window and Settings remain reachable.")
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .foregroundStyle(.secondary)
            }

            Section("Privileged Helper") {
                HStack {
                    Text(model.helperStatusTitle)
                    Spacer()
                    HelperStateBadge(snapshot: model.helperStatus)
                }

                Text(model.helperStatusDetail)
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                HStack {
                    if model.helperUsesSMAppService {
                        Button(model.helperNeedsInstall ? "Install helper" : "Uninstall helper") {
                            if model.helperNeedsInstall {
                                model.installHelper()
                            } else {
                                model.uninstallHelper()
                            }
                        }
                        .disabled(model.helperActionInProgress)
                    }

                    Button("Refresh status", action: model.refreshHelperStatus)
                        .disabled(model.helperActionInProgress)

                    if model.helperUsesSMAppService {
                        Button("Open Login Items", action: model.openHelperSettings)
                    }
                }

                if model.helperActionInProgress {
                    ProgressView()
                        .controlSize(.small)
                }

                if model.helperStatusDisplayDetail != model.helperStatusDetail {
                    Text(model.helperStatusDisplayDetail)
                        .font(.system(size: 11, weight: .medium, design: .rounded))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Section("Current Resolution") {
                Text(model.resolvedBinaryDisplay)
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .textSelection(.enabled)

                Text(model.statusEndpointDisplay)
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)

                Text(model.activeLogLocation)
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }

            Section("Notes") {
                Text("Proxy mode is managed directly by the app. TUN mode is installed as a launchd daemon, and the app reads live client status from the local HTTP endpoint instead of relying only on console output.")
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct DashboardCard<Content: View>: View {
    let title: String?
    @ViewBuilder let content: Content

    init(title: String? = nil, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let title {
                Text(title)
                    .font(.system(size: 16, weight: .bold, design: .rounded))
            }

            content
        }
        .padding(22)
        .background(
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .fill(Color.white.opacity(0.08))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .strokeBorder(Color.white.opacity(0.10), lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.12), radius: 18, x: 0, y: 14)
    }
}

private struct StateBadge: View {
    let state: ConnectionState

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: state.symbolName)
            Text(state.title)
        }
        .font(.system(size: 11, weight: .bold, design: .rounded))
        .foregroundStyle(state.tint)
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(
            Capsule(style: .continuous)
                .fill(state.tint.opacity(0.15))
        )
    }
}

private struct HelperStateBadge: View {
    let snapshot: HelperServiceSnapshot

    private var tint: Color {
        switch snapshot.state {
        case .ready:
            ConnectionState.connected.tint
        case .approvalRequired:
            ConnectionState.starting.tint
        case .notRegistered, .fallback:
            Color(red: 0.55, green: 0.62, blue: 0.72)
        case .notFound, .unknown:
            ConnectionState.error.tint
        }
    }

    var body: some View {
        Text(snapshot.title)
            .font(.system(size: 10, weight: .bold, design: .rounded))
            .foregroundStyle(tint)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                Capsule(style: .continuous)
                    .fill(tint.opacity(0.15))
            )
    }
}

private struct SectionLabel: View {
    let text: String

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        Text(text)
            .font(.system(size: 12, weight: .bold, design: .rounded))
            .textCase(.uppercase)
            .foregroundStyle(.secondary)
    }
}

private struct KeyValueRow: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 11, weight: .bold, design: .rounded))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)

            Text(value)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

private struct LabeledPicker<Value: Hashable & Identifiable>: View {
    let title: String
    @Binding var selection: Value
    let values: [Value]
    let titleKeyPath: KeyPath<Value, String>

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel(title)

            Picker(title, selection: $selection) {
                ForEach(values) { value in
                    Text(value[keyPath: titleKeyPath]).tag(value)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(width: 200, alignment: .leading)
        }
        .frame(width: 200, alignment: .leading)
    }
}

private struct LiveUplinkRow: View {
    let uplink: ClientUplinkSnapshot
    let tint: Color

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Circle()
                .fill(tint)
                .frame(width: 8, height: 8)
                .padding(.top, 5)

            VStack(alignment: .leading, spacing: 4) {
                Text(uplink.interface ?? uplink.id)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))

                Text("\(uplink.id) · \(uplink.remoteAddr)")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)

                Text(String(format: "%@ / %@ · RTT %.1f ms · Loss %.1f%% · BW %.1f Mbps", uplink.state, uplink.health, uplink.rttMs, uplink.lossPercent, uplink.bandwidthMbps))
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 0)
        }
        .padding(.vertical, 3)
    }
}

private struct InterfaceStatusRow: View {
    let interface: NetworkInterface
    let selected: Bool

    var body: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(selected ? ConnectionState.connected.tint : Color.secondary.opacity(0.35))
                .frame(width: 8, height: 8)

            VStack(alignment: .leading, spacing: 2) {
                Text(interface.displayName)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                Text("\(interface.name) \u{00B7} \(interface.kind)")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 0)
        }
        .padding(.vertical, 3)
    }
}

private struct NotesEditor: View {
    @Binding var text: String
    let prompt: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            TextField(prompt, text: $text)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .frame(maxWidth: .infinity, alignment: .leading)

            Text("Use comma, semicolon, or newline separators.")
                .font(.system(size: 11, weight: .medium, design: .rounded))
                .foregroundStyle(.secondary)
        }
    }
}

private struct PortField: View {
    let title: String
    @Binding var value: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionLabel(title)
            TextField(title, value: $value, format: .number)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12, weight: .medium, design: .monospaced))

            Text("Set to 0 to disable")
                .font(.system(size: 11, weight: .medium, design: .rounded))
                .foregroundStyle(.secondary)
        }
        .frame(width: 180, alignment: .leading)
    }
}
