//! Routing table management for TUN tunnel.
//!
//! Manages system routes to direct traffic through the TUN interface.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Route configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteConfig {
    /// Set up routes for all traffic (full tunnel).
    #[serde(default)]
    pub full_tunnel: bool,

    /// Specific networks to route through the tunnel.
    #[serde(default)]
    pub include_routes: Vec<String>,

    /// Networks to exclude from the tunnel (bypass).
    #[serde(default)]
    pub exclude_routes: Vec<String>,

    /// DNS servers to use (route DNS traffic through tunnel).
    #[serde(default)]
    pub dns_servers: Vec<IpAddr>,

    /// Whether to modify the default route.
    #[serde(default)]
    pub modify_default_route: bool,

    /// Routing table ID for policy routing (Linux).
    #[serde(default = "default_table_id")]
    pub table_id: u32,

    /// Mark for policy routing (Linux).
    #[serde(default)]
    pub fwmark: Option<u32>,
}

fn default_table_id() -> u32 {
    100
}

impl Default for RouteConfig {
    fn default() -> Self {
        Self {
            full_tunnel: false,
            include_routes: Vec::new(),
            exclude_routes: Vec::new(),
            dns_servers: Vec::new(),
            modify_default_route: false,
            table_id: default_table_id(),
            fwmark: None,
        }
    }
}

/// A network route.
#[derive(Debug, Clone)]
pub struct Route {
    /// Destination network (CIDR notation).
    pub destination: String,
    /// Gateway (optional).
    pub gateway: Option<IpAddr>,
    /// Interface name.
    pub interface: String,
    /// Metric/priority.
    pub metric: Option<u32>,
}

impl Route {
    /// Create a new route through an interface.
    pub fn via_interface(destination: &str, interface: &str) -> Self {
        Self {
            destination: destination.to_string(),
            gateway: None,
            interface: interface.to_string(),
            metric: None,
        }
    }

    /// Create a new route via a gateway.
    pub fn via_gateway(destination: &str, gateway: IpAddr, interface: &str) -> Self {
        Self {
            destination: destination.to_string(),
            gateway: Some(gateway),
            interface: interface.to_string(),
            metric: None,
        }
    }

    /// Set the metric.
    pub fn with_metric(mut self, metric: u32) -> Self {
        self.metric = Some(metric);
        self
    }
}

/// Route manager for system routing table manipulation.
pub struct RouteManager {
    /// Configuration.
    config: RouteConfig,
    
    /// TUN interface name.
    interface: String,

    /// Routes we've added (for cleanup).
    added_routes: Vec<Route>,

    /// Original default gateway (for restoration).
    original_default_gw: Option<(IpAddr, String)>,

    /// Whether routes have been set up.
    is_setup: bool,
}

impl RouteManager {
    /// Create a new route manager.
    pub fn new(config: RouteConfig, interface: String) -> Self {
        Self {
            config,
            interface,
            added_routes: Vec::new(),
            original_default_gw: None,
            is_setup: false,
        }
    }

    /// Set up all routes according to configuration.
    pub fn setup(&mut self) -> Result<()> {
        if self.is_setup {
            return Ok(());
        }

        // Save original default gateway before any modifications
        if self.config.full_tunnel || self.config.modify_default_route {
            self.original_default_gw = self.get_default_gateway()?;
            tracing::info!(
                gateway = ?self.original_default_gw,
                "Saved original default gateway"
            );
        }

        // Add excluded routes first (these bypass the tunnel)
        let exclude_routes = self.config.exclude_routes.clone();
        let original_gw = self.original_default_gw.clone();
        for network in &exclude_routes {
            if let Some((gw, iface)) = &original_gw {
                self.add_route_via_gateway(network, *gw, iface)?;
            }
        }

        // Set up full tunnel or specific routes
        if self.config.full_tunnel {
            self.setup_full_tunnel()?;
        } else {
            // Add specific routes
            for network in &self.config.include_routes.clone() {
                self.add_route(network)?;
            }
        }

        // Add DNS routes
        for dns in &self.config.dns_servers.clone() {
            let network = format!("{}/32", dns);
            self.add_route(&network)?;
        }

        self.is_setup = true;
        Ok(())
    }

    /// Tear down all routes and restore original state.
    pub fn teardown(&mut self) -> Result<()> {
        if !self.is_setup {
            return Ok(());
        }

        // Remove routes in reverse order
        let routes_to_remove: Vec<_> = self.added_routes.drain(..).collect();
        for route in routes_to_remove.into_iter().rev() {
            if let Err(e) = self.remove_route_impl(&route) {
                tracing::warn!(
                    route = %route.destination,
                    error = %e,
                    "Failed to remove route"
                );
            }
        }

        // Restore original default gateway
        if let Some((gw, iface)) = self.original_default_gw.take() {
            self.restore_default_gateway(gw, &iface)?;
        }

        self.is_setup = false;
        Ok(())
    }

    /// Add a route through the TUN interface.
    pub fn add_route(&mut self, destination: &str) -> Result<()> {
        let route = Route::via_interface(destination, &self.interface);
        self.add_route_impl(&route)?;
        self.added_routes.push(route);
        Ok(())
    }

    /// Add a route via a specific gateway.
    pub fn add_route_via_gateway(&mut self, destination: &str, gateway: IpAddr, interface: &str) -> Result<()> {
        let route = Route::via_gateway(destination, gateway, interface);
        self.add_route_impl(&route)?;
        self.added_routes.push(route);
        Ok(())
    }

    /// Check if routes are set up.
    pub fn is_setup(&self) -> bool {
        self.is_setup
    }

    // Platform-specific implementations

    fn setup_full_tunnel(&mut self) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            self.setup_full_tunnel_linux()
        }

        #[cfg(target_os = "macos")]
        {
            self.setup_full_tunnel_macos()
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            Err(Error::Config("Full tunnel not supported on this platform".into()))
        }
    }

    #[cfg(target_os = "linux")]
    fn setup_full_tunnel_linux(&mut self) -> Result<()> {
        // Use policy routing to avoid breaking connectivity to server
        let table = self.config.table_id.to_string();

        // Add default route to our table
        let output = Command::new("ip")
            .args(["route", "add", "default", "dev", &self.interface, "table", &table])
            .output()
            .map_err(|e| Error::Io(e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.contains("File exists") {
                return Err(Error::Config(format!("Failed to add default route: {}", stderr)));
            }
        }

        // Add rule to use our table
        if let Some(mark) = self.config.fwmark {
            let output = Command::new("ip")
                .args(["rule", "add", "fwmark", &mark.to_string(), "table", &table])
                .output()
                .map_err(|e| Error::Io(e))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if !stderr.contains("File exists") {
                    return Err(Error::Config(format!("Failed to add routing rule: {}", stderr)));
                }
            }
        } else {
            // Use from all rule
            let output = Command::new("ip")
                .args(["rule", "add", "from", "all", "table", &table, "priority", "100"])
                .output()
                .map_err(|e| Error::Io(e))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if !stderr.contains("File exists") {
                    return Err(Error::Config(format!("Failed to add routing rule: {}", stderr)));
                }
            }
        }

        self.added_routes.push(Route {
            destination: "default".to_string(),
            gateway: None,
            interface: self.interface.clone(),
            metric: None,
        });

        tracing::info!(
            interface = %self.interface,
            table = table,
            "Set up full tunnel routing (Linux)"
        );

        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn setup_full_tunnel_macos(&mut self) -> Result<()> {
        // On macOS, use two /1 routes to override default without replacing it
        // This is the standard VPN technique: 0.0.0.0/1 and 128.0.0.0/1
        
        // Route for 0.0.0.0/1 (lower half)
        let output = Command::new("route")
            .args(["-n", "add", "-net", "0.0.0.0/1", "-interface", &self.interface])
            .output()
            .map_err(|e| Error::Io(e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.contains("File exists") {
                return Err(Error::Config(format!("Failed to add route 0.0.0.0/1: {}", stderr)));
            }
        }

        self.added_routes.push(Route::via_interface("0.0.0.0/1", &self.interface));

        // Route for 128.0.0.0/1 (upper half)
        let output = Command::new("route")
            .args(["-n", "add", "-net", "128.0.0.0/1", "-interface", &self.interface])
            .output()
            .map_err(|e| Error::Io(e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.contains("File exists") {
                return Err(Error::Config(format!("Failed to add route 128.0.0.0/1: {}", stderr)));
            }
        }

        self.added_routes.push(Route::via_interface("128.0.0.0/1", &self.interface));

        tracing::info!(
            interface = %self.interface,
            "Set up full tunnel routing (macOS)"
        );

        Ok(())
    }

    fn add_route_impl(&self, route: &Route) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            self.add_route_linux(route)
        }

        #[cfg(target_os = "macos")]
        {
            self.add_route_macos(route)
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            Err(Error::Config("Route management not supported".into()))
        }
    }

    #[cfg(target_os = "linux")]
    fn add_route_linux(&self, route: &Route) -> Result<()> {
        let mut args = vec!["route", "add", &route.destination];

        if let Some(gw) = route.gateway {
            args.push("via");
            args.push(&gw.to_string());
        }

        args.push("dev");
        args.push(&route.interface);

        if let Some(metric) = route.metric {
            args.push("metric");
            args.push(&metric.to_string());
        }

        let output = Command::new("ip")
            .args(&args)
            .output()
            .map_err(|e| Error::Io(e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.contains("File exists") {
                return Err(Error::Config(format!("Failed to add route: {}", stderr)));
            }
        }

        tracing::debug!(destination = %route.destination, interface = %route.interface, "Added route");
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn add_route_macos(&self, route: &Route) -> Result<()> {
        let gw_str = route.gateway.map(|gw| gw.to_string());
        let mut args = vec!["-n", "add", "-net", &route.destination];

        if let Some(ref gw) = gw_str {
            args.push("-gateway");
            args.push(gw);
        } else {
            args.push("-interface");
            args.push(&route.interface);
        }

        let output = Command::new("route")
            .args(&args)
            .output()
            .map_err(|e| Error::Io(e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.contains("File exists") {
                return Err(Error::Config(format!("Failed to add route: {}", stderr)));
            }
        }

        tracing::debug!(destination = %route.destination, interface = %route.interface, "Added route");
        Ok(())
    }

    fn remove_route_impl(&self, route: &Route) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            self.remove_route_linux(route)
        }

        #[cfg(target_os = "macos")]
        {
            self.remove_route_macos(route)
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            Ok(())
        }
    }

    #[cfg(target_os = "linux")]
    fn remove_route_linux(&self, route: &Route) -> Result<()> {
        let output = Command::new("ip")
            .args(["route", "del", &route.destination, "dev", &route.interface])
            .output()
            .map_err(|e| Error::Io(e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.contains("No such process") {
                return Err(Error::Config(format!("Failed to remove route: {}", stderr)));
            }
        }

        tracing::debug!(destination = %route.destination, "Removed route");
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn remove_route_macos(&self, route: &Route) -> Result<()> {
        let output = Command::new("route")
            .args(["-n", "delete", "-net", &route.destination])
            .output()
            .map_err(|e| Error::Io(e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.contains("not in table") {
                return Err(Error::Config(format!("Failed to remove route: {}", stderr)));
            }
        }

        tracing::debug!(destination = %route.destination, "Removed route");
        Ok(())
    }

    fn get_default_gateway(&self) -> Result<Option<(IpAddr, String)>> {
        #[cfg(target_os = "linux")]
        {
            self.get_default_gateway_linux()
        }

        #[cfg(target_os = "macos")]
        {
            self.get_default_gateway_macos()
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            Ok(None)
        }
    }

    #[cfg(target_os = "linux")]
    fn get_default_gateway_linux(&self) -> Result<Option<(IpAddr, String)>> {
        let output = Command::new("ip")
            .args(["route", "show", "default"])
            .output()
            .map_err(|e| Error::Io(e))?;

        if !output.status.success() {
            return Ok(None);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        // Parse: default via 192.168.1.1 dev eth0
        for line in stdout.lines() {
            if line.starts_with("default") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 && parts[1] == "via" {
                    if let Ok(gw) = parts[2].parse::<IpAddr>() {
                        if parts[3] == "dev" {
                            return Ok(Some((gw, parts[4].to_string())));
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    #[cfg(target_os = "macos")]
    fn get_default_gateway_macos(&self) -> Result<Option<(IpAddr, String)>> {
        let output = Command::new("route")
            .args(["-n", "get", "default"])
            .output()
            .map_err(|e| Error::Io(e))?;

        if !output.status.success() {
            return Ok(None);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut gateway = None;
        let mut interface = None;

        for line in stdout.lines() {
            let line = line.trim();
            if line.starts_with("gateway:") {
                gateway = line.split(':').nth(1).and_then(|s| s.trim().parse().ok());
            } else if line.starts_with("interface:") {
                interface = line.split(':').nth(1).map(|s| s.trim().to_string());
            }
        }

        match (gateway, interface) {
            (Some(gw), Some(iface)) => Ok(Some((gw, iface))),
            _ => Ok(None),
        }
    }

    fn restore_default_gateway(&self, gateway: IpAddr, interface: &str) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            // On Linux with policy routing, just remove the rule and route from our table
            let table = self.config.table_id.to_string();

            // Remove rule
            let _ = Command::new("ip")
                .args(["rule", "del", "table", &table])
                .output();

            // Remove route from table
            let _ = Command::new("ip")
                .args(["route", "del", "default", "table", &table])
                .output();

            tracing::info!("Restored default routing (Linux)");
        }

        #[cfg(target_os = "macos")]
        {
            // On macOS, the /1 routes will be removed by the normal cleanup
            tracing::info!("Restored default routing (macOS)");
        }

        Ok(())
    }
}

impl Drop for RouteManager {
    fn drop(&mut self) {
        if self.is_setup {
            if let Err(e) = self.teardown() {
                tracing::error!(error = %e, "Failed to teardown routes on drop");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_config_default() {
        let config = RouteConfig::default();
        assert!(!config.full_tunnel);
        assert_eq!(config.table_id, 100);
    }

    #[test]
    fn test_route_creation() {
        let route = Route::via_interface("10.0.0.0/8", "tun0");
        assert_eq!(route.destination, "10.0.0.0/8");
        assert_eq!(route.interface, "tun0");
        assert!(route.gateway.is_none());

        let route = Route::via_gateway(
            "192.168.0.0/16",
            IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            "eth0"
        ).with_metric(100);
        
        assert!(route.gateway.is_some());
        assert_eq!(route.metric, Some(100));
    }
}
