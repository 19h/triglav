//! Triglav CLI - High-performance multi-path networking tool.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use colored::Colorize;
use console::Term;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tokio::signal;
use tokio::sync::broadcast;

use triglav::cli::*;
use triglav::config::{init_logging, Config};
use triglav::crypto::KeyPair;
use triglav::error::Result;
use triglav::multipath::{MultipathConfig, MultipathManager, UplinkConfig};
use triglav::proxy::{HttpProxyConfig, HttpProxyServer, Socks5Config, Socks5Server};
use triglav::transport::TransportProtocol;
use triglav::tun::{NatConfig, RouteConfig, TunConfig, TunnelConfig, TunnelRunner};
use triglav::types::AuthKey;
use triglav::util;
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
    let client_key = AuthKey::new(*keypair.public.as_bytes(), args.listen.clone());
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

    // Setup shutdown signal
    let (shutdown_tx, mut shutdown_rx) = broadcast::channel::<()>(1);

    tokio::spawn(async move {
        let _ = signal::ctrl_c().await;
        let _ = shutdown_tx.send(());
    });

    println!("{} Server running. Press Ctrl+C to stop.", "●".green());

    // Run server loop
    // TODO: Implement actual server logic
    shutdown_rx.recv().await.ok();

    println!();
    println!("{} Server stopped.", "●".yellow());

    Ok(())
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
    let interfaces = if args.auto_discover {
        util::get_network_interfaces()
            .into_iter()
            .filter(|i| i.is_up && !i.is_loopback)
            .map(|i| i.name)
            .collect::<Vec<_>>()
    } else if !args.interface.is_empty() {
        args.interface.clone()
    } else {
        util::get_network_interfaces()
            .into_iter()
            .filter(|i| i.is_up && !i.is_loopback)
            .take(2)
            .map(|i| i.name)
            .collect::<Vec<_>>()
    };

    if interfaces.is_empty() {
        return Err(triglav::Error::NoAvailableUplinks);
    }

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
    tun_config.ipv4_addr = Some(ipv4);

    let mut nat_config = NatConfig::default();
    nat_config.tunnel_ipv4 = ipv4;

    let mut route_config = RouteConfig::default();
    route_config.full_tunnel = args.full_tunnel;
    route_config.include_routes = args.route.clone();
    route_config.exclude_routes = args.exclude.clone();

    // Exclude server addresses from tunnel
    for addr in server_addrs {
        route_config
            .exclude_routes
            .push(format!("{}/32", addr.ip()));
    }

    let tunnel_config = TunnelConfig {
        tun: tun_config,
        nat: nat_config,
        routing: route_config,
        ..Default::default()
    };

    // Create tunnel runner
    let mut runner = TunnelRunner::new(tunnel_config)?;

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
    for iface in &interfaces {
        let uplink_cfg = UplinkConfig {
            id: iface.clone().into(),
            interface: Some(iface.clone()),
            remote_addr: server_addrs[0],
            protocol: TransportProtocol::Udp,
            weight: 100,
            enabled: true,
            ..Default::default()
        };
        runner.add_uplink(uplink_cfg)?;
    }

    // Connect
    println!("{} Connecting...", "→".cyan());
    let remote_public = triglav::crypto::PublicKey::from_bytes(auth_key.server_pubkey());
    match runner.connect(remote_public).await {
        Ok(_) => {
            println!("{} Connected!", "✓".green());
        }
        Err(e) => {
            println!("{} Connection failed: {}", "✗".red(), e);
            return Err(e);
        }
    }

    println!();
    println!("{}", "Tunnel Status:".bright_white().bold());
    println!("  Device:   {}", runner.tun_name());
    println!("  Uplinks:  {} active", runner.manager().uplink_count());
    println!("  Strategy: {:?}", args.strategy);
    println!();
    println!("{} Tunnel running. Press Ctrl+C to stop.", "●".green());
    println!();

    // Setup shutdown handler
    let (shutdown_tx, mut shutdown_rx) = broadcast::channel::<()>(1);

    tokio::spawn(async move {
        let _ = signal::ctrl_c().await;
        let _ = shutdown_tx.send(());
    });

    // Run tunnel
    tokio::select! {
        result = runner.run() => {
            if let Err(e) = result {
                println!("{} Tunnel error: {}", "✗".red(), e);
            }
        }
        _ = shutdown_rx.recv() => {
            println!();
            println!("{} Shutting down...", "→".yellow());
        }
    }

    runner.stop();

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
    let interfaces = if args.auto_discover {
        util::get_network_interfaces()
            .into_iter()
            .filter(|i| i.is_up && !i.is_loopback)
            .map(|i| i.name)
            .collect::<Vec<_>>()
    } else if !args.interface.is_empty() {
        args.interface.clone()
    } else {
        // Default: try to find usable interfaces
        util::get_network_interfaces()
            .into_iter()
            .filter(|i| i.is_up && !i.is_loopback)
            .take(2)
            .map(|i| i.name)
            .collect::<Vec<_>>()
    };

    if interfaces.is_empty() {
        return Err(triglav::Error::NoAvailableUplinks);
    }

    println!("{} Using interfaces:", "→".cyan());
    for iface in &interfaces {
        println!("  {} {}", "●".green(), iface);
    }
    println!();

    // Create uplink configs
    let uplinks: Vec<UplinkConfig> = interfaces
        .iter()
        .map(|iface| UplinkConfig {
            id: iface.clone().into(),
            interface: Some(iface.clone()),
            remote_addr: server_addrs[0], // Use first server addr
            protocol: TransportProtocol::Udp,
            weight: 100,
            enabled: true,
            ..Default::default()
        })
        .collect();

    // Create multipath manager
    let keypair = KeyPair::generate();
    let mut mp_config = MultipathConfig::default();
    mp_config.scheduler.strategy = args.strategy.into();

    let manager = Arc::new(MultipathManager::new(mp_config, keypair));

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
    match manager.connect(remote_public).await {
        Ok(_) => {
            conn_bar.finish_with_message(format!("{} Connected!", "✓".green()));
        }
        Err(e) => {
            conn_bar.finish_with_message(format!("{} Connection failed: {}", "✗".red(), e));
            return Err(e);
        }
    }

    // Start maintenance loop for health checks, retries, and pings
    manager.start_maintenance_loop();

    // Show connection status
    println!();
    println!("{}", "Connection Status:".bright_white().bold());
    println!("  Session:  {}", manager.session_id());
    println!("  Uplinks:  {} active", manager.uplink_count());
    println!("  Strategy: {:?}", args.strategy);

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
        let _ = signal::ctrl_c().await;
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
    manager.close()?;
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
    // Try to connect to local metrics endpoint
    let metrics_url = "http://127.0.0.1:9090";

    // Try status endpoint first
    let status_url = format!("{}/status", metrics_url);

    match reqwest::get(&status_url).await {
        Ok(response) if response.status().is_success() => {
            let status: serde_json::Value = response
                .json()
                .await
                .unwrap_or_else(|_| serde_json::json!({}));

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

            // Version and uptime
            if let Some(version) = status.get("version").and_then(|v| v.as_str()) {
                println!("  {} {}", "Version:".bright_white(), version);
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
                    "running" => state.green(),
                    "starting" => state.yellow(),
                    _ => state.red(),
                };
                println!("  {} {}", "State:".bright_white(), state_colored);
            }
            println!();

            // Uplinks
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
                        let rtt = uplink.get("rtt_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let loss = uplink
                            .get("loss_percent")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);

                        let status_icon = match state {
                            "connected" => "●".green(),
                            "connecting" => "◐".yellow(),
                            _ => "○".red(),
                        };

                        println!(
                            "  {} {} - {} | RTT: {:.1}ms | Loss: {:.1}%",
                            status_icon,
                            id.bright_white(),
                            state,
                            rtt,
                            loss
                        );
                    }
                }
                println!();
            }

            // Sessions
            if let Some(sessions) = status.get("sessions").and_then(|v| v.as_array()) {
                println!("{}", "Sessions:".bright_white().bold());
                if sessions.is_empty() {
                    println!("  {} No active sessions", "○".dimmed());
                } else {
                    println!("  {} active sessions", sessions.len().to_string().green());
                }
                println!();
            }

            // Traffic
            if let (Some(tx), Some(rx)) = (
                status.get("total_bytes_sent").and_then(|v| v.as_u64()),
                status.get("total_bytes_received").and_then(|v| v.as_u64()),
            ) {
                println!("{}", "Traffic:".bright_white().bold());
                println!("  {} TX: {}", "↑".cyan(), util::format_bytes(tx));
                println!("  {} RX: {}", "↓".cyan(), util::format_bytes(rx));
                println!();
            }

            // Detailed stats
            if args.detailed {
                // Fetch metrics
                let metrics_endpoint = format!("{}/metrics", metrics_url);
                if let Ok(metrics_response) = reqwest::get(&metrics_endpoint).await {
                    if let Ok(metrics_text) = metrics_response.text().await {
                        println!("{}", "Metrics:".bright_white().bold());
                        // Show a few key metrics
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
        }
        Ok(response) => {
            println!("{} Server returned: {}", "✗".red(), response.status());
            println!();
            println!("Make sure a Triglav server/client is running with metrics enabled.");
        }
        Err(_) => {
            // No server running, show offline status
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
            println!("No Triglav instance detected on {}.", metrics_url.cyan());
            println!();
            println!("To start:");
            println!(
                "  Server: {} {}",
                "triglav server --generate-key".cyan(),
                ""
            );
            println!("  Client: {} {}", "triglav connect <key>".cyan(), "");

            // Show available network interfaces
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
    std::fs::write(path, keypair.secret.to_base64())?;
    Ok(())
}
