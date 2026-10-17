//! Standalone entry point for the production dual-stack exit service.
use clap::Parser;
use std::{path::PathBuf, sync::Arc};
use triglav::{
    crypto::{KeyPair, SecretKey},
    server::{serve_exit_status, ExitConfig, ExitServer},
    Result,
};

#[derive(Parser)]
#[command(name = "triglav-server", about = "Multi-uplink single-exit IP service")]
struct Args {
    #[arg(long, env = "TRIGLAV_ACCESS_TOKEN", hide_env_values = true)]
    access_token: Option<String>,
    #[arg(long, default_value = "1340", value_parser = clap::value_parser!(u16).range(1280..=1367))]
    mtu: u16,
    #[arg(short, long, default_value = "0.0.0.0:7443")]
    listen: std::net::SocketAddr,
    #[arg(short, long, default_value = "127.0.0.1:9090")]
    metrics: std::net::SocketAddr,
    #[arg(short, long)]
    key: Option<PathBuf>,
    #[arg(long)]
    generate_key: bool,
    #[arg(long, default_value = "tg-exit")]
    tun_name: String,
    #[arg(long, default_value = "10.0.85.254")]
    tun_ipv4: std::net::Ipv4Addr,
    #[arg(long, default_value = "fd00:7472:6967::ffff")]
    tun_ipv6: std::net::Ipv6Addr,
    #[arg(long)]
    ipv4_only: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    triglav::config::init_logging(&triglav::config::LoggingConfig::default())?;
    let keypair = match args.key {
        Some(path) if path.exists() => KeyPair::from_secret(SecretKey::from_base64(
            std::fs::read_to_string(path)?.trim(),
        )?),
        path if args.generate_key => {
            let keypair = KeyPair::generate();
            if let Some(path) = path {
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
            }
            keypair
        }
        _ => {
            return Err(triglav::Error::Config(
                "Provide --key or --generate-key".into(),
            ))
        }
    };
    let mut config = ExitConfig::default();
    config.listen = args.listen;
    config.access_token = triglav::server::parse_access_token(args.access_token.as_deref())?;
    config.tun.name = args.tun_name;
    config.tun.mtu = args.mtu;
    config.tun.ipv4_addr = Some(args.tun_ipv4);
    config.tun.ipv6_addr = if args.ipv4_only {
        None
    } else {
        Some(args.tun_ipv6)
    };
    let server = Arc::new(ExitServer::new(config, keypair.clone()).await?);
    println!(
        "{}",
        triglav::types::AuthKey::new(
            *keypair.public.as_bytes(),
            triglav::util::advertised_server_addrs(&[server.local_addr()?])
        )
    );
    tokio::select! {
        result = server.run() => result?,
        result = serve_exit_status(Arc::clone(&server), args.metrics) => result?,
        _ = triglav::server::shutdown_signal() => server.shutdown(),
    }
    Ok(())
}
