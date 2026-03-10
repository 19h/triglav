import Foundation

final class MenuBarState: ObservableObject {
    @Published private(set) var connectionState: ConnectionState = .disconnected
    @Published private(set) var modeTitle: String = ConnectionMode.tun.title

    var canConnect: Bool {
        switch connectionState {
        case .disconnected, .error:
            true
        case .starting, .connected, .stopping:
            false
        }
    }

    var canDisconnect: Bool {
        switch connectionState {
        case .starting, .connected, .stopping:
            true
        case .disconnected, .error:
            false
        }
    }

    func update(connectionState: ConnectionState, modeTitle: String) {
        if self.connectionState != connectionState {
            self.connectionState = connectionState
        }

        if self.modeTitle != modeTitle {
            self.modeTitle = modeTitle
        }
    }
}
