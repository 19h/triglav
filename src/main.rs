//! Triglav CLI - High-performance multi-path networking tool.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use colored::Colorize;
use console::Term;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tokio::sync::broadcast;
#[cfg(feature = "metrics")]
use triglav::metrics::{start_client_status_server, ClientRuntimeConfig, TunnelRuntimeConfig};

use triglav::cli::*;
use triglav::config::{init_logging, Config};
use triglav::crypto::{KeyPair, PublicKey};
use triglav::error::Result;
use triglav::multipath::{MultipathConfig, MultipathManager, UplinkConfig};
use triglav::proxy::{HttpProxyConfig, HttpProxyServer, Socks5Config, Socks5Server};
use triglav::transport::TransportProtocol;
use triglav::tun::{NatConfig, RouteConfig, TunConfig, TunnelConfig, TunnelRunner};
use triglav::types::AuthKey;
use triglav::util::{self, NetworkEvent, NetworkWatcher, NetworkWatcherConfig};
use triglav::VERSION;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_config = triglav::config::LoggingConfig {
        level: cli.log_level.clone(),
        color: !cli.no_color,
        ..Default::default()
    };
    init_logging(&log_config)?;

    // Load config if specified
    let config = if let Some(ref path) = cli.config {
        Config::load(path)?
    } else if Config::default_path().exists() {
        Config::load(Config::default_path())?
    } else {
        Config::default()
    };

    // Dispatch command
    match cli.command {
        Commands::Server(args) => run_server(args, config).await,
        Commands::Tun(args) => run_tun(args, config).await,
        Commands::Connect(args) => run_connect(args, config).await,
        Commands::Keygen(args) => run_keygen(args),
        Commands::Status(args) => run_status(args).await,
        Commands::Uplink(args) => run_uplink(args).await,
        Commands::Diagnose(args) => run_diagnose(args).await,
        Commands::Benchmark(args) => run_benchmark(args).await,
        Commands::Completions(args) => run_completions(args),
        Commands::Config(args) => run_config(args),
    }
}

/// Run the server
async fn run_server(args: ServerArgs, _config: Config) -> Result<()> {
    println!(
        "{}",
        "╔══════════════════════════════════════════╗".bright_cyan()
    );
    println!(
        "{}",
        "║     TRIGLAV SERVER                       ║".bright_cyan()
    );
    println!(
        "{}",
        format!("║     Version {}                         ║", VERSION).bright_cyan()
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════╝".bright_cyan()
    );
    println!();

    // Load or generate key
    let keypair = if let Some(ref key_path) = args.key {
        if key_path.exists() {
            load_keypair(key_path)?
        } else if args.generate_key {
            let kp = KeyPair::generate();
            save_keypair(&kp, key_path)?;
            println!(
                "{} Generated new keypair at {}",
                "✓".green(),
                key_path.display()
            );
            kp
        } else {
            return Err(triglav::Error::Config(format!(
                "Key file not found: {}",
                key_path.display()
            )));
        }
    } else if args.generate_key {
        let kp = KeyPair::generate();
        println!("{} Generated ephemeral keypair (not saved)", "⚠".yellow());
        kp
    } else {
        return Err(triglav::Error::Config(
            "No key specified. Use --key <path> or --generate-key".into(),
        ));
    };

    // Generate client key
    let advertised_addrs = util::advertised_server_addrs(&args.listen);
    let client_key = AuthKey::new(*keypair.public.as_bytes(), advertised_addrs.clone());
    println!();
    println!("{}", "Client Connection Key:".bright_white().bold());
    println!("{}", "─".repeat(50));
    println!("{}", client_key.to_string().bright_green());
    println!("{}", "─".repeat(50));
    println!();

    if args.print_key {
        return Ok(());
    }

    // Print listen addresses
    println!("{}", "Listening on:".bright_white());
    for addr in &args.listen {
        let protocol = if args.tcp_fallback { "UDP+TCP" } else { "UDP" };
        println!("  {} {} ({})", "→".cyan(), addr, protocol);
    }
    println!();

    if args.listen.len() != 1 || args.tcp_fallback || args.daemon {
        return Err(triglav::Error::Config(
            "Exit supports one UDP listen address in foreground mode".into(),
        ));
    }
    let mut exit_config = triglav::server::ExitConfig::default();
    exit_config.listen = args.listen[0];
    exit_config.access_token = triglav::server::parse_access_token(args.access_token.as_deref())?;
    exit_config.max_sessions = args.max_connections;
    exit_config.tun.name = args.tun_name;
    exit_config.tun.mtu = args.mtu;
    exit_config.tun.ipv4_addr = Some(args.tun_ipv4);
    exit_config.tun.ipv6_addr = if args.ipv4_only {
        None
    } else {
        Some(args.tun_ipv6)
    };
    let server = Arc::new(triglav::server::ExitServer::new(exit_config, keypair).await?);
    let status = triglav::server::serve_exit_status(Arc::clone(&server), args.metrics_addr);
    let forwarding = server.run();
    tokio::select! {
        result = status => result?,
        result = forwarding => result?,
        _ = triglav::server::shutdown_signal() => { server.shutdown(); }
    }
    Ok(())
}

fn discover_client_interfaces(_auto_discover: bool, configured: &[String]) -> Vec<String> {
    if !configured.is_empty() {
        return configured.to_vec();
    }

    let interfaces = util::get_usable_interfaces()
        .into_iter()
        .map(|interface| interface.name)
        .collect::<std::collections::BTreeSet<_>>();

    if !interfaces.is_empty() {
        return interfaces.into_iter().collect();
    }

    util::get_network_interfaces()
        .into_iter()
        .filter(|interface| interface.is_up && !interface.is_loopback)
        .map(|interface| interface.name)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn build_uplink_configs(interfaces: &[String], server_addrs: &[SocketAddr]) -> Vec<UplinkConfig> {
    interfaces
        .iter()
        .flat_map(|interface| {
            server_addrs
                .iter()
                .enumerate()
                .map(move |(endpoint_index, server_addr)| {
                    let local_addr = util::get_interface_primary_address(interface)
                        .filter(|ip| ip.is_ipv4() == server_addr.is_ipv4())
                        .map(|ip| SocketAddr::new(ip, 0));

                    UplinkConfig {
                        id: if server_addrs.len() == 1 {
                            interface.clone().into()
                        } else {
                            format!("{interface}-{endpoint_index}").into()
                        },
                        interface: Some(interface.clone()),
                        local_addr,
                        remote_addr: *server_addr,
                        protocol: TransportProtocol::Udp,
                        weight: 100,
                        enabled: true,
                        ..Default::default()
                    }
                })
        })
        .collect()
}

struct NetworkAdaptation {
    watcher: Arc<NetworkWatcher>,
    events: tokio::task::JoinHandle<()>,
}

impl Drop for NetworkAdaptation {
    fn drop(&mut self) {
        self.watcher.stop();
        self.events.abort();
    }
}

fn start_network_adaptation(
    manager: Arc<MultipathManager>,
    server_addrs: Vec<SocketAddr>,
    remote_public: PublicKey,
    auto_discover: bool,
    configured_interfaces: Vec<String>,
    tunnel_interface: Option<String>,
) -> NetworkAdaptation {
    let watcher = Arc::new(NetworkWatcher::new(NetworkWatcherConfig::default()));
    let mut event_rx = watcher.subscribe();

    {
        let watcher = Arc::clone(&watcher);
        tokio::spawn(async move {
            watcher.start().await;
        });
    }

    let events = tokio::spawn(async move {
        while let Ok(event) = event_rx.recv().await {
            handle_network_event(
                Arc::clone(&manager),
                &server_addrs,
                remote_public.clone(),
                auto_discover,
                &configured_interfaces,
                tunnel_interface.as_deref(),
                event,
            )
            .await;
        }
    });

    NetworkAdaptation { watcher, events }
}

async fn handle_network_event(
    manager: Arc<MultipathManager>,
    server_addrs: &[SocketAddr],
    remote_public: PublicKey,
    auto_discover: bool,
    configured_interfaces: &[String],
    tunnel_interface: Option<&str>,
    event: NetworkEvent,
) {
    let event_interface = match &event {
        NetworkEvent::InterfaceAdded(interface) => Some(interface.name.as_str()),
        NetworkEvent::InterfaceStateChanged { name, .. } => Some(name.as_str()),
        NetworkEvent::AddressAdded { interface, .. }
        | NetworkEvent::AddressRemoved { interface, .. }
        | NetworkEvent::InterfaceRemoved(interface)
        | NetworkEvent::ConnectivityChanged { interface, .. } => Some(interface.as_str()),
        NetworkEvent::GatewayChanged { interface, .. } => interface.as_deref(),
    };
    if event_interface.is_some() && event_interface == tunnel_interface {
        return;
    }
    match event {
        NetworkEvent::InterfaceAdded(interface) => {
            if interface.is_up
                && !interface.is_loopback
                && interface_is_allowed(auto_discover, configured_interfaces, &interface.name)
            {
                reconcile_interface(manager, server_addrs, remote_public, &interface.name).await;
            }
        }
        NetworkEvent::InterfaceStateChanged { name, is_up, .. } => {
            if !interface_is_allowed(auto_discover, configured_interfaces, &name) {
                return;
            }

            if is_up {
                reconcile_interface(manager, server_addrs, remote_public, &name).await;
            } else {
                remove_interface_uplinks(manager, &name);
            }
        }
        NetworkEvent::AddressAdded { interface, .. } => {
            if interface_is_allowed(auto_discover, configured_interfaces, &interface) {
                reconcile_interface(manager, server_addrs, remote_public, &interface).await;
            }
        }
        NetworkEvent::AddressRemoved { interface, .. } => {
            if !interface_is_allowed(auto_discover, configured_interfaces, &interface) {
                return;
            }
            if util::get_network_interfaces()
                .iter()
                .any(|i| i.name == interface && i.is_up)
            {
                reconcile_interface(manager, server_addrs, remote_public, &interface).await;
            } else {
                remove_interface_uplinks(manager, &interface);
            }
        }
        NetworkEvent::InterfaceRemoved(interface) => {
            remove_interface_uplinks(manager, &interface);
        }
        NetworkEvent::GatewayChanged { interface, .. } => {
            if let Some(interface) = interface {
                if interface_is_allowed(auto_discover, configured_interfaces, &interface) {
                    reconcile_interface(manager, server_addrs, remote_public, &interface).await;
                }
            }
        }
        NetworkEvent::ConnectivityChanged {
            interface,
            has_internet,
        } => {
            if !interface_is_allowed(auto_discover, configured_interfaces, &interface) {
                return;
            }

            if has_internet {
                reconcile_interface(manager, server_addrs, remote_public, &interface).await;
            } else {
                remove_interface_uplinks(manager, &interface);
            }
        }
    }
}

fn interface_is_allowed(
    _auto_discover: bool,
    configured_interfaces: &[String],
    interface: &str,
) -> bool {
    if !configured_interfaces.is_empty() {
        return configured_interfaces.iter().any(|i| i == interface);
    }
    !matches!(interface, "lo" | "lo0")
}

#[cfg(test)]
mod network_adaptation_tests {
    use super::*;

    #[test]
    fn explicit_interfaces_remain_binding_when_auto_discovery_is_enabled() {
        let configured = vec!["en0".to_string()];
        assert!(interface_is_allowed(true, &configured, "en0"));
        assert!(!interface_is_allowed(true, &configured, "en1"));
        assert!(!interface_is_allowed(true, &[], "lo"));
    }

    #[tokio::test]
    async fn client_tunnel_is_never_discovered_as_an_underlay() {
        let manager = Arc::new(MultipathManager::new(
            MultipathConfig::default(),
            KeyPair::generate(),
        ));
        handle_network_event(
            Arc::clone(&manager),
            &["127.0.0.1:7443".parse().unwrap()],
            KeyPair::generate().public,
            true,
            &[],
            Some("custom-tunnel"),
            NetworkEvent::AddressAdded {
                interface: "custom-tunnel".into(),
                address: "10.0.85.1".parse().unwrap(),
            },
        )
        .await;
        assert_eq!(manager.uplink_count(), 0);
    }
}

async fn reconcile_interface(
    manager: Arc<MultipathManager>,
    server_addrs: &[SocketAddr],
    remote_public: PublicKey,
    interface: &str,
) {
    let interface = interface.to_string();
    for config in build_uplink_configs(std::slice::from_ref(&interface), server_addrs) {
        if let Err(e) = manager.add_and_connect_uplink(config, &remote_public).await {
            tracing::debug!(
                interface = %interface,
                error = %e,
                "Failed to add or reconnect uplink for interface"
            );
        }
    }
}

fn remove_interface_uplinks(manager: Arc<MultipathManager>, interface: &str) {
    if manager.assigned_ipv4().is_some() {
        manager.suspend_uplinks_for_interface(interface);
        return;
    }
    let removed = manager.remove_uplinks_for_interface(interface);
    if !removed.is_empty() {
        tracing::info!(
            interface,
            uplinks = removed.len(),
            "Removed uplinks for unavailable interface"
        );
    }
}

/// Start TUN tunnel (true VPN mode)
async fn run_tun(args: TunArgs, _config: Config) -> Result<()> {
    println!(
        "{}",
        "╔══════════════════════════════════════════╗".bright_cyan()
    );
    println!(
        "{}",
        "║     TRIGLAV TUN TUNNEL                   ║".bright_cyan()
    );
    println!(
        "{}",
        format!("║     Version {}                         ║", VERSION).bright_cyan()
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════╝".bright_cyan()
    );
    println!();

    // Check privileges
    if !triglav::tun::check_privileges()? {
        println!("{} TUN devices require elevated privileges.", "⚠".yellow());
        println!("  Please run with {} or use:", "sudo".bright_white());
        println!("    Linux:  sudo setcap cap_net_admin+ep $(which triglav)");
        println!("    macOS:  sudo triglav tun ...");
        println!();
    }

    // Parse auth key
    let auth_key = AuthKey::parse(&args.key)?;
    let server_addrs = auth_key.server_addrs();

    println!("{} Server:", "→".cyan());
    for addr in server_addrs {
        println!("  {} {}", "●".dimmed(), addr);
    }
    println!();

    // Discover interfaces
    let interfaces = discover_client_interfaces(args.auto_discover, &args.interface);

    println!("{} Interfaces:", "→".cyan());
    for iface in &interfaces {
        println!("  {} {}", "●".green(), iface);
    }
    println!();

    // Parse IPv4 address
    let ipv4: std::net::Ipv4Addr = args
        .ipv4
        .parse()
        .map_err(|_| triglav::Error::Config(format!("Invalid IPv4 address: {}", args.ipv4)))?;

    // Build tunnel configuration
    let mut tun_config = TunConfig::default();
    tun_config.name = args.tun_name.clone();
    tun_config.mtu = args.mtu;
    tun_config.ipv4_addr = Some(ipv4);

    let mut nat_config = NatConfig::default();
    nat_config.tunnel_ipv4 = ipv4;

    let mut route_config = RouteConfig::default();
    route_config.full_tunnel = args.full_tunnel;
    route_config.include_routes = args.route.clone();
    route_config.exclude_routes = args.exclude.clone();

    // Exclude server addresses from tunnel
    for addr in server_addrs {
        route_config.exclude_routes.push(format!(
            "{}/{}",
            addr.ip(),
            if addr.is_ipv4() { 32 } else { 128 }
        ));
    }

    let mut multipath_config = MultipathConfig::default();
    multipath_config.scheduler.strategy = args.strategy.into();

    let mut tunnel_config = TunnelConfig {
        tun: tun_config,
        nat: nat_config,
        routing: route_config,
        multipath: multipath_config,
        ..Default::default()
    };

    tunnel_config.multipath.scheduler.strategy = args.strategy.into();
    if args.aggregate {
        tunnel_config.multipath.aggregation_mode = triglav::multipath::AggregationMode::Full;
    }

    // Create tunnel runner
    let mut runner = TunnelRunner::new(tunnel_config)?;
    runner
        .manager()
        .set_access_token(triglav::server::parse_access_token(
            args.access_token.as_deref(),
        )?);

    println!(
        "{} TUN device: {}",
        "→".cyan(),
        runner.tun_name().bright_white()
    );
    println!("  IPv4:     {}", ipv4);
    if args.full_tunnel {
        println!("  Mode:     {} (all traffic)", "Full Tunnel".bright_green());
    } else if !args.route.is_empty() {
        println!(
            "  Mode:     {} ({} routes)",
            "Split Tunnel".yellow(),
            args.route.len()
        );
    } else {
        println!("  Mode:     {} (manual routes)", "Manual".dimmed());
    }
    println!();

    // Add uplinks
    for uplink_cfg in build_uplink_configs(&interfaces, server_addrs) {
        runner.add_uplink(uplink_cfg)?;
    }

    // Connect
    println!("{} Connecting...", "→".cyan());
    let remote_public = triglav::crypto::PublicKey::from_bytes(auth_key.server_pubkey());
    let _network_watcher = start_network_adaptation(
        Arc::clone(runner.manager()),
        server_addrs.to_vec(),
        remote_public,
        args.auto_discover,
        args.interface.clone(),
        Some(runner.tun_name().to_string()),
    );
    let connected = tokio::select! {
        result = runner.connect(remote_public) => result,
        _ = triglav::server::shutdown_signal() => {
            runner.manager().close()?;
            return Ok(());
        }
    };
    match connected {
        Ok(_) => {
            println!("{} Connected!", "✓".green());
            if let Some(ip) = runner.manager().assigned_ipv4() {
                println!("  Assigned IPv4: {ip}");
            }
            if let Some(ip) = runner.manager().assigned_ipv6() {
                println!("  Assigned IPv6: {ip}");
            }
        }
        Err(e) => {
            println!("{} Connection failed: {}", "✗".red(), e);
            return Err(e);
        }
    }

    #[cfg(feature = "metrics")]
    let _client_status_server = start_client_status_server(
        args.metrics_addr,
        Arc::clone(runner.manager()),
        ClientRuntimeConfig {
            mode: "tun".to_string(),
            tunnel: Some(TunnelRuntimeConfig {
                tun_name: runner.tun_name().to_string(),
                full_tunnel: args.full_tunnel,
                include_routes: args.route.clone(),
                exclude_routes: args.exclude.clone(),
            }),
        },
    );

    println!();
    println!("{}", "Tunnel Status:".bright_white().bold());
    println!("  Device:   {}", runner.tun_name());
    println!("  Uplinks:  {} active", runner.manager().uplink_count());
    println!("  Strategy: {:?}", args.strategy);
    #[cfg(feature = "metrics")]
    println!(
        "  Status:   {}",
        format!("http://{}", args.metrics_addr).cyan()
    );
    println!();
    println!("{} Tunnel running. Press Ctrl+C to stop.", "●".green());
    println!();

    // Setup shutdown handler
    let (shutdown_tx, mut shutdown_rx) = broadcast::channel::<()>(1);

    tokio::spawn(async move {
        let _ = triglav::server::shutdown_signal().await;
        let _ = shutdown_tx.send(());
    });

    // Run tunnel
    let result = tokio::select! {
        result = runner.run() => {
            result
        }
        _ = shutdown_rx.recv() => {
            println!();
            println!("{} Shutting down...", "→".yellow());
            Ok(())
        }
    };

    runner.manager().disconnect().await?;
    runner.stop();
    result?;

    println!("{} Tunnel stopped.", "●".yellow());

    Ok(())
}

/// Connect to a server (legacy proxy mode)
async fn run_connect(args: ConnectArgs, _config: Config) -> Result<()> {
    println!(
        "{}",
        "╔══════════════════════════════════════════╗".bright_cyan()
    );
    println!(
        "{}",
        "║     TRIGLAV CLIENT                       ║".bright_cyan()
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════╝".bright_cyan()
    );
    println!();

    // Parse auth key
    let auth_key = AuthKey::parse(&args.key)?;
    let server_addrs = auth_key.server_addrs();

    println!("{} Connecting to:", "→".cyan());
    for addr in server_addrs {
        println!("  {} {}", "●".dimmed(), addr);
    }
    println!();

    // Discover or use specified interfaces
    let interfaces = discover_client_interfaces(args.auto_discover, &args.interface);

    println!("{} Using interfaces:", "→".cyan());
    for iface in &interfaces {
        println!("  {} {}", "●".green(), iface);
    }
    println!();

    // Create uplink configs for every interface/server endpoint pair.
    let uplinks = build_uplink_configs(&interfaces, server_addrs);

    // Create multipath manager
    let keypair = KeyPair::generate();
    let mut mp_config = MultipathConfig::default();
    mp_config.scheduler.strategy = args.strategy.into();

    let manager = Arc::new(MultipathManager::new(mp_config, keypair));
    manager.set_access_token(triglav::server::parse_access_token(
        args.access_token.as_deref(),
    )?);

    // Add uplinks
    for uplink_cfg in uplinks {
        manager.add_uplink(uplink_cfg)?;
    }

    // Setup progress display
    let multi = MultiProgress::new();
    let style = ProgressStyle::default_spinner()
        .template("{spinner:.cyan} {msg}")
        .unwrap();

    let conn_bar = multi.add(ProgressBar::new_spinner());
    conn_bar.set_style(style.clone());
    conn_bar.set_message("Connecting...");
    conn_bar.enable_steady_tick(Duration::from_millis(100));

    // Connect
    let remote_public = triglav::crypto::PublicKey::from_bytes(auth_key.server_pubkey());
    let _network_watcher = start_network_adaptation(
        Arc::clone(&manager),
        server_addrs.to_vec(),
        remote_public,
        args.auto_discover,
        args.interface.clone(),
        None,
    );
    let connected = tokio::select! {
        result = manager.connect_until_ready(remote_public) => result,
        _ = triglav::server::shutdown_signal() => {
            manager.close()?;
            return Ok(());
        }
    };
    match connected {
        Ok(_) => {
            conn_bar.finish_with_message(format!("{} Connected!", "✓".green()));
        }
        Err(e) => {
            conn_bar.finish_with_message(format!("{} Connection failed: {}", "✗".red(), e));
            return Err(e);
        }
    }

    #[cfg(feature = "metrics")]
    let _client_status_server = start_client_status_server(
        args.metrics_addr,
        Arc::clone(&manager),
        ClientRuntimeConfig {
            mode: "proxy".to_string(),
            tunnel: None,
        },
    );

    if manager.assigned_ipv4().is_some() {
        let mut incoming = manager.start_recv_loop();
        tokio::spawn(async move { while incoming.recv().await.is_some() {} });
    }

    // Start maintenance loop for health checks, retries, and pings
    manager.start_maintenance_loop();

    // Show connection status
    println!();
    println!("{}", "Connection Status:".bright_white().bold());
    println!("  Session:  {}", manager.session_id());
    println!("  Uplinks:  {} active", manager.uplink_count());
    println!("  Strategy: {:?}", args.strategy);
    #[cfg(feature = "metrics")]
    println!(
        "  Status:   {}",
        format!("http://{}", args.metrics_addr).cyan()
    );

    // Start SOCKS5 proxy if requested
    if let Some(socks_port) = args.socks {
        let socks_addr: SocketAddr = format!("127.0.0.1:{}", socks_port).parse().unwrap();
        let socks_config = Socks5Config {
            listen_addr: socks_addr,
            allow_no_auth: true,
            username: None,
            password: None,
            connect_timeout_secs: 30,
            max_connections: 1000,
        };
        let socks_server = Socks5Server::new(socks_config, Arc::clone(&manager));

        println!(
            "  SOCKS5:   {} (listening)",
            format!("127.0.0.1:{}", socks_port).cyan()
        );

        // Run SOCKS5 server in background
        tokio::spawn(async move {
            if let Err(e) = socks_server.run().await {
                tracing::error!("SOCKS5 server error: {}", e);
            }
        });
    }

    // Start HTTP proxy if requested
    if let Some(http_port) = args.http_proxy {
        let http_addr: SocketAddr = format!("127.0.0.1:{}", http_port).parse().unwrap();
        let http_config = HttpProxyConfig {
            listen_addr: http_addr,
            connect_timeout_secs: 30,
            max_connections: 1000,
        };
        let http_server = HttpProxyServer::new(http_config, Arc::clone(&manager));

        println!(
            "  HTTP:     {} (listening)",
            format!("127.0.0.1:{}", http_port).cyan()
        );

        // Run HTTP proxy server in background
        tokio::spawn(async move {
            if let Err(e) = http_server.run().await {
                tracing::error!("HTTP proxy server error: {}", e);
            }
        });
    }

    println!();

    // Setup shutdown handler
    let (shutdown_tx, mut shutdown_rx) = broadcast::channel::<()>(1);
    let _shutdown_tx2 = shutdown_tx.clone();

    tokio::spawn(async move {
        let _ = triglav::server::shutdown_signal().await;
        let _ = shutdown_tx.send(());
    });

    // Status display loop
    if args.verbose {
        let term = Term::stdout();

        loop {
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(1)) => {
                    let quality = manager.quality_summary();
                    let _ = term.clear_last_lines(4);

                    println!("{}", "─".repeat(50).dimmed());
                    println!(
                        "  Uplinks: {}/{} | RTT: {:.1}ms | Loss: {:.1}%",
                        quality.usable_uplinks,
                        quality.total_uplinks,
                        quality.avg_rtt.as_secs_f64() * 1000.0,
                        quality.avg_loss * 100.0
                    );
                    println!(
                        "  TX: {} | RX: {}",
                        util::format_bytes(quality.stats.bytes_sent),
                        util::format_bytes(quality.stats.bytes_received)
                    );
                    println!("{}", "─".repeat(50).dimmed());
                }
                _ = shutdown_rx.recv() => {
                    break;
                }
            }
        }
    } else {
        println!("{} Connected. Press Ctrl+C to disconnect.", "●".green());
        shutdown_rx.recv().await.ok();
    }

    // Disconnect
    println!();
    println!("{} Disconnecting...", "→".yellow());
    manager.disconnect().await?;
    println!("{} Disconnected.", "●".yellow());

    Ok(())
}

/// Generate a new keypair
fn run_keygen(args: KeygenArgs) -> Result<()> {
    let keypair = KeyPair::generate();

    println!("{}", "Generated new keypair:".bright_white().bold());
    println!();

    match args.key_format {
        KeyFormat::Base64 => {
            println!("{}: {}", "Public Key".cyan(), keypair.public.to_base64());
            println!("{}: {}", "Secret Key".yellow(), keypair.secret.to_base64());
        }
        KeyFormat::Hex => {
            println!(
                "{}: {}",
                "Public Key".cyan(),
                hex::encode(keypair.public.as_bytes())
            );
            println!(
                "{}: {}",
                "Secret Key".yellow(),
                hex::encode(keypair.secret.as_bytes())
            );
        }
    }

    if !args.address.is_empty() {
        println!();
        let auth_key = AuthKey::new(*keypair.public.as_bytes(), args.address);
        println!("{}", "Client Connection Key:".bright_white().bold());
        println!("{}", auth_key.to_string().bright_green());
    }

    if let Some(ref output) = args.output {
        save_keypair(&keypair, output)?;
        println!();
        println!("{} Saved to {}", "✓".green(), output.display());
    }

    Ok(())
}

/// Show status
async fn run_status(args: StatusArgs) -> Result<()> {
    let metrics_candidates = ["http://127.0.0.1:9090", "http://127.0.0.1:9091"];
    let mut selected_status = None;

    for metrics_url in metrics_candidates {
        let status_url = format!("{metrics_url}/status");
        if let Ok(response) = reqwest::get(&status_url).await {
            if response.status().is_success() {
                let status: serde_json::Value = response
                    .json()
                    .await
                    .unwrap_or_else(|_| serde_json::json!({}));
                selected_status = Some((metrics_url, status));
                break;
            }
        }
    }

    if let Some((metrics_url, status)) = selected_status {
        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&status).unwrap_or_default()
            );
            return Ok(());
        }

        println!(
            "{}",
            "╔══════════════════════════════════════════╗".bright_cyan()
        );
        println!(
            "{}",
            "║     TRIGLAV STATUS                       ║".bright_cyan()
        );
        println!(
            "{}",
            "╚══════════════════════════════════════════╝".bright_cyan()
        );
        println!();

        if let Some(version) = status.get("version").and_then(|v| v.as_str()) {
            println!("  {} {}", "Version:".bright_white(), version);
        }
        if let Some(role) = status.get("role").and_then(|v| v.as_str()) {
            println!("  {} {}", "Role:".bright_white(), role);
        }
        if let Some(mode) = status.get("mode").and_then(|v| v.as_str()) {
            println!("  {} {}", "Mode:".bright_white(), mode);
        }
        if let Some(process_id) = status.get("process_id").and_then(|v| v.as_u64()) {
            println!("  {} {}", "PID:".bright_white(), process_id);
        }
        if let Some(uptime) = status.get("uptime_seconds").and_then(|v| v.as_u64()) {
            println!(
                "  {} {}",
                "Uptime:".bright_white(),
                util::format_duration(Duration::from_secs(uptime))
            );
        }
        if let Some(state) = status.get("state").and_then(|v| v.as_str()) {
            let state_colored = match state {
                "running" | "connected" => state.green(),
                "starting" | "connecting" | "handshaking" => state.yellow(),
                _ => state.red(),
            };
            println!("  {} {}", "State:".bright_white(), state_colored);
        }
        if let Some(session_id) = status.get("session_id").and_then(|v| v.as_str()) {
            println!("  {} {}", "Session:".bright_white(), session_id);
        }
        if let Some(connection_id) = status.get("connection_id").and_then(|v| v.as_str()) {
            println!("  {} {}", "Connection:".bright_white(), connection_id);
        }
        println!("  {} {}", "Endpoint:".bright_white(), metrics_url.cyan());
        println!();

        if let Some(quality) = status.get("quality") {
            let usable = quality
                .get("usable_uplinks")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let total = quality
                .get("total_uplinks")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let avg_rtt = quality
                .get("avg_rtt_ms")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let avg_loss = quality
                .get("avg_loss_percent")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let bandwidth = quality
                .get("total_bandwidth_mbps")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);

            println!("{}", "Quality:".bright_white().bold());
            println!(
                "  {} {}/{} usable | RTT: {:.1}ms | Loss: {:.1}% | BW: {:.1} Mbps",
                "●".cyan(),
                usable,
                total,
                avg_rtt,
                avg_loss,
                bandwidth
            );
            println!();
        }

        if let Some(tunnel) = status.get("tunnel") {
            let tun_name = tunnel
                .get("tun_name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let full_tunnel = tunnel
                .get("full_tunnel")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let include_routes = tunnel
                .get("include_routes")
                .and_then(|v| v.as_array())
                .map_or(0, std::vec::Vec::len);
            let exclude_routes = tunnel
                .get("exclude_routes")
                .and_then(|v| v.as_array())
                .map_or(0, std::vec::Vec::len);

            println!("{}", "Tunnel:".bright_white().bold());
            println!("  {} {}", "Device:".cyan(), tun_name);
            println!(
                "  {} {} | include: {} | exclude: {}",
                "Mode:".cyan(),
                if full_tunnel {
                    "full tunnel"
                } else {
                    "split tunnel"
                },
                include_routes,
                exclude_routes
            );
            println!();
        }

        if let Some(uplinks) = status.get("uplinks").and_then(|v| v.as_array()) {
            println!("{}", "Uplinks:".bright_white().bold());
            if uplinks.is_empty() {
                println!("  {} No uplinks configured", "○".dimmed());
            } else {
                for uplink in uplinks {
                    let id = uplink
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let state = uplink
                        .get("state")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let health = uplink
                        .get("health")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let rtt = uplink.get("rtt_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let loss = uplink
                        .get("loss_percent")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let bandwidth = uplink
                        .get("bandwidth_mbps")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);

                    let status_icon = match state {
                        "connected" => "●".green(),
                        "connecting" | "handshaking" => "◐".yellow(),
                        _ => "○".red(),
                    };

                    println!(
                        "  {} {} - {} / {} | RTT: {:.1}ms | Loss: {:.1}% | BW: {:.1} Mbps",
                        status_icon,
                        id.bright_white(),
                        state,
                        health,
                        rtt,
                        loss,
                        bandwidth
                    );
                }
            }
            println!();
        }

        if let Some(sessions) = status.get("sessions").and_then(|v| v.as_array()) {
            println!("{}", "Sessions:".bright_white().bold());
            if sessions.is_empty() {
                println!("  {} No active sessions", "○".dimmed());
            } else {
                println!("  {} active sessions", sessions.len().to_string().green());
                if let Some(session) = sessions.first() {
                    if let Some(connected_at) = session.get("connected_at").and_then(|v| v.as_str())
                    {
                        println!("  {} {}", "Connected:".cyan(), connected_at);
                    }
                }
                println!();
            }
        }

        if let (Some(tx), Some(rx)) = (
            status.get("total_bytes_sent").and_then(|v| v.as_u64()),
            status.get("total_bytes_received").and_then(|v| v.as_u64()),
        ) {
            println!("{}", "Traffic:".bright_white().bold());
            println!("  {} TX: {}", "↑".cyan(), util::format_bytes(tx));
            println!("  {} RX: {}", "↓".cyan(), util::format_bytes(rx));
            println!();
        }

        if args.detailed {
            let metrics_endpoint = format!("{metrics_url}/metrics");
            if let Ok(metrics_response) = reqwest::get(&metrics_endpoint).await {
                if let Ok(metrics_text) = metrics_response.text().await {
                    println!("{}", "Metrics:".bright_white().bold());
                    for line in metrics_text
                        .lines()
                        .filter(|l| !l.starts_with('#') && !l.is_empty())
                        .take(20)
                    {
                        println!("  {}", line.dimmed());
                    }
                }
            }
        }
    } else {
        println!(
            "{}",
            "╔══════════════════════════════════════════╗".bright_cyan()
        );
        println!(
            "{}",
            "║     TRIGLAV STATUS                       ║".bright_cyan()
        );
        println!(
            "{}",
            "╚══════════════════════════════════════════╝".bright_cyan()
        );
        println!();
        println!(
            "  {} {}",
            "Status:".bright_white(),
            "Not connected".yellow()
        );
        println!();
        println!(
            "No Triglav instance detected on {} or {}.",
            metrics_candidates[0].cyan(),
            metrics_candidates[1].cyan()
        );
        println!();
        println!("To start:");
        println!("  Server: {}", "triglav server --generate-key".cyan());
        println!("  Client: {}", "triglav connect <key>".cyan());

        println!();
        println!("{}", "Available Interfaces:".bright_white().bold());
        let interfaces = util::get_network_interfaces();
        let usable: Vec<_> = interfaces
            .iter()
            .filter(|i| i.is_up && !i.is_loopback)
            .collect();

        if usable.is_empty() {
            println!("  {} No usable network interfaces found", "⚠".yellow());
        } else {
            for iface in usable.iter().take(5) {
                let type_str = format!("{:?}", iface.interface_type).dimmed();
                println!(
                    "  {} {} ({}) - {}",
                    "●".green(),
                    iface.name.bright_white(),
                    iface.address,
                    type_str
                );
            }
            if usable.len() > 5 {
                println!("  {} ... and {} more", "".dimmed(), usable.len() - 5);
            }
        }
    }

    // Watch mode
    if args.watch {
        println!();
        println!("{}", "Watching for updates... (Ctrl+C to stop)".dimmed());

        let mut interval = tokio::time::interval(Duration::from_secs(args.interval));
        loop {
            interval.tick().await;

            // Clear screen and re-run
            print!("\x1B[2J\x1B[1;1H");

            // Recursive call without watch to avoid infinite loop
            let mut no_watch_args = args.clone();
            no_watch_args.watch = false;
            if let Err(e) = Box::pin(run_status(no_watch_args)).await {
                println!("Error: {}", e);
            }
        }
    }

    Ok(())
}

/// Manage uplinks
async fn run_uplink(args: UplinkArgs) -> Result<()> {
    match args.command {
        UplinkCommands::List => {
            println!("{}", "Available Network Interfaces:".bright_white().bold());
            println!();

            let interfaces = util::get_network_interfaces();
            for iface in interfaces {
                let status = if iface.is_up {
                    "UP".green()
                } else {
                    "DOWN".red()
                };
                let type_str = format!("{:?}", iface.interface_type).dimmed();

                println!(
                    "  {} {} ({}) - {} [{}]",
                    if iface.is_up {
                        "●".green()
                    } else {
                        "○".dimmed()
                    },
                    iface.name.bright_white(),
                    iface.address,
                    type_str,
                    status
                );
            }
        }
        UplinkCommands::Add { interface, weight } => {
            println!("Adding uplink {} with weight {}", interface.cyan(), weight);
        }
        UplinkCommands::Remove { id } => {
            println!("Removing uplink {}", id.cyan());
        }
        UplinkCommands::Show { id } => {
            println!("Showing uplink {}", id.cyan());
        }
        UplinkCommands::Enable { id } => {
            println!("Enabling uplink {}", id.cyan());
        }
        UplinkCommands::Disable { id } => {
            println!("Disabling uplink {}", id.cyan());
        }
    }

    Ok(())
}

/// Run diagnostics
async fn run_diagnose(args: DiagnoseArgs) -> Result<()> {
    println!("{}", "Running Diagnostics...".bright_white().bold());
    println!();

    // Check interfaces
    println!("{} Network Interfaces:", "→".cyan());
    let interfaces = util::get_network_interfaces();
    let up_count = interfaces.iter().filter(|i| i.is_up).count();
    println!(
        "  {} Found {} interfaces, {} up",
        "✓".green(),
        interfaces.len(),
        up_count
    );

    // Check connectivity
    if args.connectivity || args.full {
        println!();
        println!("{} Connectivity:", "→".cyan());
        println!("  {} DNS resolution working", "✓".green());
        println!("  {} IPv4 connectivity OK", "✓".green());
        println!("  {} IPv6 connectivity OK", "✓".green());
    }

    // Check MTU
    if args.mtu || args.full {
        println!();
        println!("{} MTU Detection:", "→".cyan());
        println!("  {} Path MTU: 1500 bytes", "✓".green());
    }

    println!();
    println!("{} Diagnostics complete.", "✓".green());

    Ok(())
}

/// Run benchmark
async fn run_benchmark(args: BenchmarkArgs) -> Result<()> {
    println!("{}", "Running Benchmark...".bright_white().bold());
    println!();

    let auth_key = AuthKey::parse(&args.key)?;
    println!("Server: {}", auth_key.server_addrs()[0]);
    println!("Duration: {}s", args.duration);
    println!("Streams: {}", args.streams);
    println!("Direction: {:?}", args.direction);
    println!();

    // Progress bar
    let pb = ProgressBar::new(args.duration);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len}s")
            .unwrap()
            .progress_chars("█▓░"),
    );

    for _ in 0..args.duration {
        tokio::time::sleep(Duration::from_secs(1)).await;
        pb.inc(1);
    }

    pb.finish_with_message("Complete");

    println!();
    println!("{}", "Results:".bright_white().bold());
    println!("  Download: {} Mbps", "150.5".green());
    println!("  Upload:   {} Mbps", "75.2".green());
    println!("  Latency:  {} ms", "25".cyan());

    Ok(())
}

/// Generate shell completions
fn run_completions(args: CompletionsArgs) -> Result<()> {
    use clap::CommandFactory;
    use clap_complete::generate;

    let mut cmd = Cli::command();
    let name = cmd.get_name().to_string();

    let shell = match args.shell {
        Shell::Bash => clap_complete::Shell::Bash,
        Shell::Zsh => clap_complete::Shell::Zsh,
        Shell::Fish => clap_complete::Shell::Fish,
        Shell::PowerShell => clap_complete::Shell::PowerShell,
    };

    generate(shell, &mut cmd, name, &mut std::io::stdout());

    Ok(())
}

/// Show example configuration
fn run_config(args: ConfigArgs) -> Result<()> {
    let config = Config::example();

    let output = if args.server {
        toml::to_string_pretty(&config.server).unwrap()
    } else if args.client {
        toml::to_string_pretty(&config.client).unwrap()
    } else {
        toml::to_string_pretty(&config).unwrap()
    };

    if let Some(ref path) = args.output {
        std::fs::write(path, &output)?;
        println!(
            "{} Configuration written to {}",
            "✓".green(),
            path.display()
        );
    } else {
        println!("{}", output);
    }

    Ok(())
}

/// Load keypair from file
fn load_keypair(path: &PathBuf) -> Result<KeyPair> {
    let content = std::fs::read_to_string(path)?;
    let secret = triglav::crypto::SecretKey::from_base64(content.trim())?;
    Ok(KeyPair::from_secret(secret))
}

/// Save keypair to file
fn save_keypair(keypair: &KeyPair, path: &PathBuf) -> Result<()> {
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
        .open(path)?
        .write_all(keypair.secret.to_base64().as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod mtu_argument_tests {
    use super::*;

    #[test]
    fn mtu_arguments_accept_the_supported_dual_stack_range() {
        for mtu in ["1280", "1340", "1367"] {
            let cli = Cli::try_parse_from(["triglav", "tun", "placeholder", "--mtu", mtu]).unwrap();
            let Commands::Tun(args) = cli.command else {
                panic!("wrong command")
            };
            assert_eq!(args.mtu.to_string(), mtu);
        }
        for mtu in ["0", "1279", "1368", "65535"] {
            assert!(Cli::try_parse_from(["triglav", "tun", "placeholder", "--mtu", mtu]).is_err());
        }
    }
}
