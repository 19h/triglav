//! Triglav standalone server binary.

use std::net::SocketAddr;
use std::path::PathBuf;

use tracing::{info, warn};

use triglav::config::{init_logging, ServerConfig};
use triglav::crypto::{KeyPair, SecretKey};
use triglav::error::{Error, Result};
use triglav::server::{daemonize, DaemonConfig, PidFileGuard, ServerRuntimeOptions};
use triglav::types::AuthKey;

#[derive(Debug)]
struct Args {
    listen_addr: SocketAddr,
    metrics_addr: SocketAddr,
    key_path: Option<PathBuf>,
    generate_key: bool,
    daemon_mode: bool,
    pid_file: Option<PathBuf>,
}

impl Args {
    fn parse() -> Result<Option<Self>> {
        let args: Vec<String> = std::env::args().collect();

        let mut parsed = Self {
            listen_addr: "0.0.0.0:7443".parse().unwrap(),
            metrics_addr: "0.0.0.0:9090".parse().unwrap(),
            key_path: None,
            generate_key: false,
            daemon_mode: false,
            pid_file: None,
        };

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "-l" | "--listen" => {
                    parsed.listen_addr = parse_next(&args, &mut i, "--listen")?;
                }
                "-m" | "--metrics" => {
                    parsed.metrics_addr = parse_next(&args, &mut i, "--metrics")?;
                }
                "-k" | "--key" => {
                    parsed.key_path = Some(PathBuf::from(next_arg(&args, &mut i, "--key")?));
                }
                "--generate-key" => {
                    parsed.generate_key = true;
                }
                "-d" | "--daemon" => {
                    parsed.daemon_mode = true;
                }
                "--pid-file" => {
                    parsed.pid_file = Some(PathBuf::from(next_arg(&args, &mut i, "--pid-file")?));
                }
                "-h" | "--help" => {
                    print_help();
                    return Ok(None);
                }
                unknown => {
                    return Err(Error::Config(format!("Unknown argument: {unknown}")));
                }
            }
            i += 1;
        }

        Ok(Some(parsed))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let Some(args) = Args::parse()? else {
        return Ok(());
    };

    let _pid_guard = if args.daemon_mode {
        let daemon_config = DaemonConfig {
            pid_file: args.pid_file.clone(),
            work_dir: PathBuf::from("/"),
            user: None,
            group: None,
            umask: Some(0o027),
            close_fds: true,
        };

        daemonize(&daemon_config)?;
        args.pid_file
            .as_ref()
            .map(|path| PidFileGuard::new(path).expect("Failed to create PID file"))
    } else {
        None
    };

    init_logging(&triglav::config::LoggingConfig::default())?;

    let keypair = load_or_generate_keypair(args.key_path.as_ref(), args.generate_key)?;

    let config = ServerConfig {
        enabled: true,
        listen_addrs: vec![args.listen_addr],
        ..Default::default()
    };

    let advertised_addrs = triglav::util::advertised_server_addrs(&[args.listen_addr]);
    let auth_key = AuthKey::new(*keypair.public.as_bytes(), advertised_addrs.clone());

    if !args.daemon_mode {
        println!();
        println!("╔══════════════════════════════════════════╗");
        println!("║     TRIGLAV SERVER                       ║");
        println!(
            "║     Version {}                         ║",
            triglav::VERSION
        );
        println!("╚══════════════════════════════════════════╝");
        println!();
        println!("Listening on: {}", args.listen_addr);
        println!("Advertised endpoints:");
        for addr in &advertised_addrs {
            println!("  {}", addr);
        }
        println!("Metrics at:   http://{}", args.metrics_addr);
        println!();
        println!("Client Connection Key:");
        println!("{}", auth_key);
        println!();
    }

    info!("Triglav server starting");
    info!(
        "Listen: {}, Metrics: {}",
        args.listen_addr, args.metrics_addr
    );

    triglav::server::run_server(ServerRuntimeOptions {
        config,
        keypair,
        metrics_addr: args.metrics_addr,
    })
    .await?;

    info!("Triglav server stopped");
    Ok(())
}

fn load_or_generate_keypair(path: Option<&PathBuf>, generate_key: bool) -> Result<KeyPair> {
    if let Some(path) = path {
        if path.exists() {
            let content = std::fs::read_to_string(path)?;
            let secret = SecretKey::from_base64(content.trim())?;
            Ok(KeyPair::from_secret(secret))
        } else if generate_key {
            let keypair = KeyPair::generate();
            std::fs::write(path, keypair.secret.to_base64())?;
            info!("Generated new keypair at {}", path.display());
            Ok(keypair)
        } else {
            Err(Error::Config(format!(
                "Key file not found: {}",
                path.display()
            )))
        }
    } else if generate_key {
        warn!("Using ephemeral keypair (not saved)");
        Ok(KeyPair::generate())
    } else {
        Err(Error::Config("No key specified".into()))
    }
}

fn parse_next<T: std::str::FromStr>(args: &[String], i: &mut usize, flag: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    let value = next_arg(args, i, flag)?;
    value
        .parse()
        .map_err(|e| Error::Config(format!("Invalid value for {flag}: {e}")))
}

fn next_arg(args: &[String], i: &mut usize, flag: &str) -> Result<String> {
    if *i + 1 >= args.len() {
        return Err(Error::Config(format!("Missing value for {flag}")));
    }
    *i += 1;
    Ok(args[*i].clone())
}

fn print_help() {
    println!("Triglav Server");
    println!();
    println!("Usage: triglav-server [OPTIONS]");
    println!();
    println!("Options:");
    println!("  -l, --listen <ADDR>    Listen address (default: 0.0.0.0:7443)");
    println!("  -m, --metrics <ADDR>   Metrics HTTP address (default: 0.0.0.0:9090)");
    println!("  -k, --key <PATH>       Path to key file");
    println!("      --generate-key     Generate new key if not exists");
    println!("  -d, --daemon           Run as daemon");
    println!("      --pid-file <PATH>  PID file path (for daemon mode)");
    println!("  -h, --help             Show this help");
}
