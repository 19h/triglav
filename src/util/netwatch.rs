//! Network interface monitoring for hotplug detection.
//!
//! Watches for interface add/remove/change events and notifies subscribers.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

use super::{get_network_interfaces, NetworkInterface};

/// Network change event.
#[derive(Debug, Clone)]
pub enum NetworkEvent {
    /// A new interface was added.
    InterfaceAdded(NetworkInterface),
    /// An interface was removed.
    InterfaceRemoved(String),
    /// An interface's state changed (up/down).
    InterfaceStateChanged {
        name: String,
        is_up: bool,
        is_running: bool,
    },
    /// An address was added to an interface.
    AddressAdded { interface: String, address: IpAddr },
    /// An address was removed from an interface.
    AddressRemoved { interface: String, address: IpAddr },
    /// Default gateway changed.
    GatewayChanged {
        interface: Option<String>,
        gateway: Option<IpAddr>,
    },
    /// Connectivity status changed.
    ConnectivityChanged {
        interface: String,
        has_internet: bool,
    },
}

/// Network watcher configuration.
#[derive(Debug, Clone)]
pub struct NetworkWatcherConfig {
    /// Polling interval for fallback polling mode.
    pub poll_interval: Duration,
    /// Whether to use native netlink (Linux) or polling.
    pub use_native_events: bool,
    /// Interfaces to ignore (e.g., docker, virbr).
    pub ignore_patterns: Vec<String>,
}

impl Default for NetworkWatcherConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(2),
            use_native_events: true,
            ignore_patterns: vec![
                "docker".to_string(),
                "virbr".to_string(),
                "veth".to_string(),
                "br-".to_string(),
            ],
        }
    }
}

/// Network watcher that monitors for interface changes.
pub struct NetworkWatcher {
    config: NetworkWatcherConfig,
    /// Current known state.
    state: Arc<RwLock<NetworkState>>,
    /// Event broadcaster.
    event_tx: broadcast::Sender<NetworkEvent>,
    /// Shutdown signal.
    shutdown_tx: broadcast::Sender<()>,
}

/// Current network state snapshot.
#[derive(Debug, Default)]
struct NetworkState {
    /// Interfaces by name.
    interfaces: HashMap<String, Vec<NetworkInterface>>,
    /// Last update time.
    last_update: Option<std::time::Instant>,
}

impl NetworkWatcher {
    /// Create a new network watcher.
    pub fn new(config: NetworkWatcherConfig) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        let (shutdown_tx, _) = broadcast::channel(1);

        Self {
            config,
            state: Arc::new(RwLock::new(NetworkState::default())),
            event_tx,
            shutdown_tx,
        }
    }

    /// Subscribe to network events.
    pub fn subscribe(&self) -> broadcast::Receiver<NetworkEvent> {
        self.event_tx.subscribe()
    }

    /// Get current interfaces.
    pub fn interfaces(&self) -> Vec<NetworkInterface> {
        let state = self.state.read();
        state.interfaces.values().flatten().cloned().collect()
    }

    /// Get interface by name.
    pub fn get_interface(&self, name: &str) -> Option<NetworkInterface> {
        let state = self.state.read();
        state.interfaces.get(name)?.first().cloned()
    }

    /// Start watching for network changes.
    pub async fn start(&self) {
        // Initial scan
        self.refresh().await;

        #[cfg(target_os = "linux")]
        if self.config.use_native_events {
            self.start_netlink_monitor().await;
        } else {
            self.start_polling().await;
        }

        #[cfg(not(target_os = "linux"))]
        self.start_polling().await;
    }

    /// Stop watching.
    pub fn stop(&self) {
        let _ = self.shutdown_tx.send(());
    }

    /// Refresh network state.
    pub async fn refresh(&self) {
        let interfaces = get_network_interfaces();
        let mut new_state: HashMap<String, Vec<NetworkInterface>> = HashMap::new();

        for iface in interfaces {
            // Skip ignored interfaces
            if self.should_ignore(&iface.name) {
                continue;
            }

            new_state.entry(iface.name.clone()).or_default().push(iface);
        }

        // Compare with old state and emit events
        let old_state = {
            let mut state = self.state.write();
            let old = std::mem::take(&mut state.interfaces);
            state.interfaces = new_state.clone();
            state.last_update = Some(std::time::Instant::now());
            old
        };

        self.emit_diff_events(&old_state, &new_state);
    }

    /// Check if an interface should be ignored.
    fn should_ignore(&self, name: &str) -> bool {
        for pattern in &self.config.ignore_patterns {
            if name.starts_with(pattern) {
                return true;
            }
        }
        false
    }

    /// Emit events for state differences.
    fn emit_diff_events(
        &self,
        old: &HashMap<String, Vec<NetworkInterface>>,
        new: &HashMap<String, Vec<NetworkInterface>>,
    ) {
        // Check for removed interfaces
        for name in old.keys() {
            if !new.contains_key(name) {
                debug!("Interface removed: {}", name);
                let _ = self
                    .event_tx
                    .send(NetworkEvent::InterfaceRemoved(name.clone()));
            }
        }

        // Check for added interfaces and state changes
        for (name, new_ifaces) in new {
            if let Some(old_ifaces) = old.get(name) {
                // Check for state changes
                let old_up = old_ifaces.iter().any(|i| i.is_up);
                let new_up = new_ifaces.iter().any(|i| i.is_up);
                let old_running = old_ifaces.iter().any(|i| i.is_running);
                let new_running = new_ifaces.iter().any(|i| i.is_running);

                if old_up != new_up || old_running != new_running {
                    debug!(
                        "Interface state changed: {} up={} running={}",
                        name, new_up, new_running
                    );
                    let _ = self.event_tx.send(NetworkEvent::InterfaceStateChanged {
                        name: name.clone(),
                        is_up: new_up,
                        is_running: new_running,
                    });
                }

                // Check for address changes
                let old_addrs: std::collections::HashSet<_> =
                    old_ifaces.iter().map(|i| i.address).collect();
                let new_addrs: std::collections::HashSet<_> =
                    new_ifaces.iter().map(|i| i.address).collect();

                for addr in new_addrs.difference(&old_addrs) {
                    debug!("Address added: {} on {}", addr, name);
                    let _ = self.event_tx.send(NetworkEvent::AddressAdded {
                        interface: name.clone(),
                        address: *addr,
                    });
                }

                for addr in old_addrs.difference(&new_addrs) {
                    debug!("Address removed: {} from {}", addr, name);
                    let _ = self.event_tx.send(NetworkEvent::AddressRemoved {
                        interface: name.clone(),
                        address: *addr,
                    });
                }
            } else {
                // New interface
                if let Some(iface) = new_ifaces.first() {
                    info!("Interface added: {}", name);
                    let _ = self
                        .event_tx
                        .send(NetworkEvent::InterfaceAdded(iface.clone()));
                }
            }
        }
    }

    /// Start polling-based monitoring.
    async fn start_polling(&self) {
        let state = Arc::clone(&self.state);
        let event_tx = self.event_tx.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let interval = self.config.poll_interval;
        let ignore_patterns = self.config.ignore_patterns.clone();

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);

            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        let interfaces = get_network_interfaces();
                        let mut new_state: HashMap<String, Vec<NetworkInterface>> = HashMap::new();

                        for iface in interfaces {
                            let should_ignore = ignore_patterns.iter()
                                .any(|p| iface.name.starts_with(p));
                            if should_ignore {
                                continue;
                            }

                            new_state.entry(iface.name.clone())
                                .or_default()
                                .push(iface);
                        }

                        // Compare and emit
                        let old_state = {
                            let mut s = state.write();
                            let old = std::mem::take(&mut s.interfaces);
                            s.interfaces = new_state.clone();
                            s.last_update = Some(std::time::Instant::now());
                            old
                        };

                        // Emit events for changes
                        for name in old_state.keys() {
                            if !new_state.contains_key(name) {
                                let _ = event_tx.send(NetworkEvent::InterfaceRemoved(name.clone()));
                            }
                        }

                        for (name, new_ifaces) in &new_state {
                            if !old_state.contains_key(name) {
                                if let Some(iface) = new_ifaces.first() {
                                    let _ = event_tx.send(NetworkEvent::InterfaceAdded(iface.clone()));
                                }
                            }
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        debug!("Network watcher polling stopped");
                        break;
                    }
                }
            }
        });
    }

    /// Start netlink-based monitoring (Linux only).
    #[cfg(target_os = "linux")]
    async fn start_netlink_monitor(&self) {
        use futures::StreamExt;
        use tracing::{debug, error, info, warn};

        let state = Arc::clone(&self.state);
        let event_tx = self.event_tx.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let ignore_patterns = self.config.ignore_patterns.clone();
        let poll_interval = self.config.poll_interval;

        tokio::spawn(async move {
            // Try to connect to netlink
            let (connection, handle, mut messages) = match rtnetlink::new_connection() {
                Ok(c) => c,
                Err(e) => {
                    error!(
                        "Failed to create netlink connection: {}, falling back to polling",
                        e
                    );
                    // Fallback to polling
                    let mut ticker = tokio::time::interval(poll_interval);
                    loop {
                        tokio::select! {
                            _ = ticker.tick() => {
                                // Simple polling refresh
                                let interfaces = get_network_interfaces();
                                let mut new_state: HashMap<String, Vec<NetworkInterface>> = HashMap::new();

                                for iface in interfaces {
                                    let should_ignore = ignore_patterns.iter()
                                        .any(|p| iface.name.starts_with(p));
                                    if should_ignore {
                                        continue;
                                    }
                                    new_state.entry(iface.name.clone())
                                        .or_default()
                                        .push(iface);
                                }

                                let mut s = state.write();
                                s.interfaces = new_state;
                                s.last_update = Some(std::time::Instant::now());
                            }
                            _ = shutdown_rx.recv() => {
                                break;
                            }
                        }
                    }
                    return;
                }
            };

            // Spawn the connection handler
            tokio::spawn(connection);

            info!("Started netlink network monitor");

            // Process netlink messages
            loop {
                tokio::select! {
                    msg = messages.next() => {
                        match msg {
                            Some((message, _)) => {
                                use netlink_packet_route::RouteNetlinkMessage;
                                use netlink_packet_core::NetlinkPayload;

                                // In rtnetlink 0.14+, payload is wrapped in NetlinkPayload
                                let NetlinkPayload::InnerMessage(inner) = message.payload else {
                                    continue;
                                };
                                match inner {
                                    RouteNetlinkMessage::NewLink(link) => {
                                        let name = link.attributes.iter()
                                            .find_map(|attr| {
                                                if let netlink_packet_route::link::LinkAttribute::IfName(n) = attr {
                                                    Some(n.clone())
                                                } else {
                                                    None
                                                }
                                            });

                                        if let Some(name) = name {
                                            if !ignore_patterns.iter().any(|p| name.starts_with(p)) {
                                                debug!("Netlink: new link {}", name);
                                                // Refresh state
                                                let interfaces = get_network_interfaces();
                                                if let Some(iface) = interfaces.iter().find(|i| i.name == name) {
                                                    let _ = event_tx.send(NetworkEvent::InterfaceAdded(iface.clone()));
                                                }
                                            }
                                        }
                                    }
                                    RouteNetlinkMessage::DelLink(link) => {
                                        let name = link.attributes.iter()
                                            .find_map(|attr| {
                                                if let netlink_packet_route::link::LinkAttribute::IfName(n) = attr {
                                                    Some(n.clone())
                                                } else {
                                                    None
                                                }
                                            });

                                        if let Some(name) = name {
                                            debug!("Netlink: del link {}", name);
                                            let _ = event_tx.send(NetworkEvent::InterfaceRemoved(name));
                                        }
                                    }
                                    RouteNetlinkMessage::NewAddress(addr) => {
                                        let iface_idx = addr.header.index;
                                        let iface_name = super::if_indextoname(iface_idx);

                                        if let Some(name) = iface_name {
                                            // Extract address - in netlink-packet-route 0.19+, Address contains IpAddr directly
                                            for attr in &addr.attributes {
                                                if let netlink_packet_route::address::AddressAttribute::Address(ip) = attr {
                                                    debug!("Netlink: new address {} on {}", ip, name);
                                                    let _ = event_tx.send(NetworkEvent::AddressAdded {
                                                        interface: name.clone(),
                                                        address: *ip,
                                                    });
                                                }
                                            }
                                        }
                                    }
                                    RouteNetlinkMessage::DelAddress(addr) => {
                                        let iface_idx = addr.header.index;
                                        let iface_name = super::if_indextoname(iface_idx);

                                        if let Some(name) = iface_name {
                                            for attr in &addr.attributes {
                                                if let netlink_packet_route::address::AddressAttribute::Address(ip) = attr {
                                                    debug!("Netlink: del address {} from {}", ip, name);
                                                    let _ = event_tx.send(NetworkEvent::AddressRemoved {
                                                        interface: name.clone(),
                                                        address: *ip,
                                                    });
                                                }
                                            }
                                        }
                                    }
                                    RouteNetlinkMessage::NewRoute(_) | RouteNetlinkMessage::DelRoute(_) => {
                                        // Route changed - could be gateway change
                                        debug!("Netlink: route change detected");
                                        // TODO: Parse and emit GatewayChanged event
                                    }
                                    _ => {}
                                }
                            }
                            None => {
                                warn!("Netlink connection closed");
                                break;
                            }
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        info!("Network watcher stopped");
                        break;
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_network_watcher_creation() {
        let watcher = NetworkWatcher::new(NetworkWatcherConfig::default());
        let _rx = watcher.subscribe();

        // Refresh should populate state
        watcher.refresh().await;

        let interfaces = watcher.interfaces();
        // Should have at least one interface on most systems
        println!("Found {} interfaces", interfaces.len());
    }

    #[test]
    fn test_should_ignore() {
        let watcher = NetworkWatcher::new(NetworkWatcherConfig::default());

        assert!(watcher.should_ignore("docker0"));
        assert!(watcher.should_ignore("virbr0"));
        assert!(watcher.should_ignore("veth123abc"));
        assert!(!watcher.should_ignore("eth0"));
        assert!(!watcher.should_ignore("wlan0"));
    }
}
