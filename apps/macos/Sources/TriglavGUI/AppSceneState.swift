import Foundation

final class AppSceneState: ObservableObject {
    private var storedShowStatusBarIcon: Bool

    var showStatusBarIcon: Bool {
        get { storedShowStatusBarIcon }
        set {
            guard storedShowStatusBarIcon != newValue else { return }
            objectWillChange.send()
            storedShowStatusBarIcon = newValue
        }
    }

    init(showStatusBarIcon: Bool = true) {
        storedShowStatusBarIcon = showStatusBarIcon
    }
}
