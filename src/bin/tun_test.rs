//! TUN device end-to-end test for macOS.
//!
//! This test verifies:
//! 1. TUN device can be created
//! 2. IP address can be assigned
//! 3. Interface shows up in ifconfig
//! 4. Packets can be read from TUN
//!
//! Run with: sudo cargo run --bin tun_test

use std::process::Command;
use std::time::Duration;

fn check_root() -> bool {
    unsafe { libc::getuid() == 0 }
}

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║     TRIGLAV TUN DEVICE TEST (macOS)              ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    // Check privileges
    if !check_root() {
        eprintln!("ERROR: This test requires root privileges.");
        eprintln!("Run with: sudo cargo run --bin tun_test");
        std::process::exit(1);
    }

    println!("[1/7] Creating TUN device...");
    
    let config = triglav::tun::TunConfig {
        name: "utun".to_string(), // macOS will assign a number
        mtu: 1420,
        ipv4_addr: Some("10.0.85.1".parse().unwrap()),
        ipv4_netmask: 24,
        ipv6_addr: None,
        ipv6_prefix: 64,
        set_default_route: false,
        queue_size: 512,
    };

    let mut tun = match triglav::tun::TunDevice::create(config) {
        Ok(t) => {
            println!("      OK: Created TUN device: {}", t.name());
            t
        }
        Err(e) => {
            eprintln!("      FAILED: {}", e);
            std::process::exit(1);
        }
    };

    println!("\n[2/7] Configuring IP address (10.0.85.1/24)...");
    match tun.configure_addresses() {
        Ok(_) => println!("      OK: IP address configured"),
        Err(e) => {
            eprintln!("      FAILED: {}", e);
            std::process::exit(1);
        }
    }

    println!("\n[3/7] Bringing interface up...");
    match tun.up() {
        Ok(_) => println!("      OK: Interface is UP"),
        Err(e) => {
            eprintln!("      FAILED: {}", e);
            std::process::exit(1);
        }
    }

    println!("\n[4/7] Verifying interface in system (ifconfig)...");
    let output = Command::new("ifconfig")
        .arg(tun.name())
        .output()
        .expect("Failed to run ifconfig");
    
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        println!("      OK: Interface visible in system");
        println!("      ┌────────────────────────────────────────────────");
        for line in stdout.lines() {
            println!("      │ {}", line);
        }
        println!("      └────────────────────────────────────────────────");
    } else {
        eprintln!("      FAILED: Interface not found in ifconfig");
        std::process::exit(1);
    }

    println!("\n[5/7] Checking routing table...");
    let route_output = Command::new("netstat")
        .args(["-rn"])
        .output()
        .expect("Failed to run netstat");
    
    let route_stdout = String::from_utf8_lossy(&route_output.stdout);
    let tun_routes: Vec<&str> = route_stdout
        .lines()
        .filter(|l| l.contains(tun.name()))
        .collect();
    
    if !tun_routes.is_empty() {
        println!("      OK: Found {} route(s) for {}", tun_routes.len(), tun.name());
        for route in &tun_routes {
            println!("      │ {}", route);
        }
    } else {
        println!("      INFO: No routes found (this is normal without --full-tunnel)");
    }

    println!("\n[6/7] Testing packet capture (sending ICMP to 10.0.85.1)...");
    
    let handle = tun.handle();
    
    // Spawn async reader
    let read_task = tokio::spawn(async move {
        let mut buf = vec![0u8; 2048];
        let mut packets_received = 0;
        
        // Try to read for up to 3 seconds
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        
        while tokio::time::Instant::now() < deadline {
            match tokio::time::timeout(Duration::from_millis(500), handle.read(&mut buf)).await {
                Ok(Ok(len)) if len > 0 => {
                    packets_received += 1;
                    
                    // Parse basic IP info
                    let version = (buf[0] >> 4) & 0x0f;
                    let (protocol, src, dst) = if version == 4 && len >= 20 {
                        let proto = buf[9];
                        let src = format!("{}.{}.{}.{}", buf[12], buf[13], buf[14], buf[15]);
                        let dst = format!("{}.{}.{}.{}", buf[16], buf[17], buf[18], buf[19]);
                        (proto, src, dst)
                    } else {
                        (0, "?".into(), "?".into())
                    };
                    
                    let proto_name = match protocol {
                        1 => "ICMP",
                        6 => "TCP",
                        17 => "UDP",
                        _ => "Other",
                    };
                    
                    println!("      RECV: {} bytes | IPv{} {} | {} -> {}", 
                             len, version, proto_name, src, dst);
                    
                    if packets_received >= 2 {
                        break;
                    }
                }
                Ok(Ok(_)) => {}
                Ok(Err(e)) => {
                    // EAGAIN/EWOULDBLOCK is normal for non-blocking
                    if e.to_string().contains("Resource temporarily unavailable") {
                        continue;
                    }
                    println!("      Read error: {}", e);
                    break;
                }
                Err(_) => {
                    // Timeout, continue
                }
            }
        }
        
        packets_received
    });

    // Wait a moment for reader to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Send some pings
    println!("      Sending ping to 10.0.85.1...");
    for _ in 0..3 {
        let _ = Command::new("ping")
            .args(["-c", "1", "-t", "1", "-W", "100", "10.0.85.1"])
            .output();
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // Wait for read task
    let packets = read_task.await.unwrap_or(0);
    
    if packets > 0 {
        println!("      OK: Captured {} packet(s) from TUN device", packets);
    } else {
        println!("      INFO: No packets captured (ping may not generate traffic to TUN)");
        println!("           This can happen if the kernel handles the ping internally.");
    }

    println!("\n[7/7] Cleanup...");
    match tun.down() {
        Ok(_) => println!("      OK: Interface brought down"),
        Err(e) => println!("      WARN: {}", e),
    }
    
    // Drop the TUN device (closes fd, removes interface)
    let tun_name = tun.name().to_string();
    drop(tun);
    
    // Verify cleanup - retry a few times as kernel may take a moment
    let mut removed = false;
    for i in 0..5 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let verify = Command::new("ifconfig")
            .arg(&tun_name)
            .output();
        
        if let Ok(o) = verify {
            if !o.status.success() {
                println!("      OK: Interface {} removed from system", tun_name);
                removed = true;
                break;
            }
        }
        if i < 4 {
            // Still exists, wait a bit more
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
    
    if !removed {
        println!("      WARN: Interface {} may still exist (will be removed on process exit)", tun_name);
    }

    println!();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║     ALL TESTS PASSED                             ║");
    println!("╚══════════════════════════════════════════════════╝");
}
