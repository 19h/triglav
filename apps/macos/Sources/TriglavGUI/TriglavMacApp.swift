import SwiftUI

@main
struct TriglavMacApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var sceneState: AppSceneState
    @StateObject private var menuBarState: MenuBarState
    private let model: AppModel

    init() {
        let sceneState = AppSceneState()
        let menuBarState = MenuBarState()
        _sceneState = StateObject(wrappedValue: sceneState)
        _menuBarState = StateObject(wrappedValue: menuBarState)
        model = AppModel(sceneState: sceneState, menuBarState: menuBarState)
    }

    var body: some Scene {
        Window("Triglav", id: "main") {
            MainWindowView()
                .environmentObject(model)
                .frame(minWidth: 720, minHeight: 480)
        }
        .defaultSize(width: 900, height: 600)

        Settings {
            SettingsView()
                .environmentObject(model)
                .frame(width: 520)
        }

        MenuBarExtra("Triglav", systemImage: "bolt.horizontal.circle.fill", isInserted: .init(
            get: { sceneState.showStatusBarIcon },
            set: { sceneState.showStatusBarIcon = $0 }
        )) {
            MenuBarContentView(
                state: menuBarState,
                connectAction: model.connect,
                disconnectAction: model.disconnect,
                quitAction: model.quitApp
            )
        }
    }
}
