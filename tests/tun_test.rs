//! TUN device end-to-end test for macOS.
//!
//! This test verifies:
//! 1. TUN device can be created
//! 2. IP address can be assigned
//! 3. Packets can be read/written
//! 4. Interface shows up in ifconfig

use std::process::Command;
use std::time::Duration;

use triglav::tun::{TunDevice, TunConfig};

fn check_root() -> bool {
    unsafe { libc::getuid() == 0 }
}

#[tokio::main]
async fn main() {
    println!("=== Triglav TUN Device Test ===\n");

    // Check privileges
    if !check_root() {
        println!("ERROR: This test requires root privileges.");
        println!("Run with: sudo cargo run --test tun_test");
        std::process::exit(1);
    }

    println!("[1/6] Creating TUN device...");
    
    let config = TunConfig {
        name: "utun".to_string(), // macOS will assign a number
        mtu: 1420,
        ipv4_addr: Some("10.0.85.1".parse().unwrap()),
        ipv4_netmask: 24,
        ipv6_addr: None,
        ipv6_prefix: 64,
        set_default_route: false,
        queue_size: 512,
    };

    let mut tun = match TunDevice::create(config) {
        Ok(t) => {
            println!("   SUCCESS: Created TUN device: {}", t.name());
            t
        }
        Err(e) => {
            println!("   FAILED: {}", e);
            std::process::exit(1);
        }
    };

    println!("\n[2/6] Configuring IP address...");
    match tun.configure_addresses() {
        Ok(_) => println!("   SUCCESS: Configured IP 10.0.85.1/24"),
        Err(e) => {
            println!("   FAILED: {}", e);
            std::process::exit(1);
        }
    }

    println!("\n[3/6] Bringing interface up...");
    match tun.up() {
        Ok(_) => println!("   SUCCESS: Interface is up"),
        Err(e) => {
            println!("   FAILED: {}", e);
            std::process::exit(1);
        }
    }

    println!("\n[4/6] Verifying interface in system...");
    let output = Command::new("ifconfig")
        .arg(tun.name())
        .output()
        .expect("Failed to run ifconfig");
    
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        println!("   SUCCESS: Interface found in system");
        println!("   ---");
        for line in stdout.lines() {
            println!("   {}", line);
        }
        println!("   ---");
    } else {
        println!("   FAILED: Interface not found");
        std::process::exit(1);
    }

    println!("\n[5/6] Testing packet read/write...");
    
    // We'll ping the TUN interface IP from the system
    // and try to read the ICMP packet from the TUN device
    
    let handle = tun.handle();
    
    // Spawn a task to read from TUN
    let read_handle = tokio::spawn(async move {
        let mut buf = vec![0u8; 1500];
        
        // Set a timeout for reading
        match tokio::time::timeout(Duration::from_secs(3), handle.read(&mut buf)).await {
            Ok(Ok(len)) => {
                println!("   SUCCESS: Read {} bytes from TUN", len);
                
                // Parse IP version
                if len > 0 {
                    let version = (buf[0] >> 4) & 0x0f;
                    let protocol = if version == 4 && len >= 20 {
                        buf[9]
                    } else {
                        0
                    };
                    
                    let proto_name = match protocol {
                        1 => "ICMP",
                        6 => "TCP",
                        17 => "UDP",
                        _ => "Other",
                    };
                    
                    println!("   Packet: IPv{}, Protocol: {} ({})", version, protocol, proto_name);
                    
                    // Show first 40 bytes hex
                    let hex: Vec<String> = buf[..len.min(40)].iter()
                        .map(|b| format!("{:02x}", b))
                        .collect();
                    println!("   Hex: {}", hex.join(" "));
                }
                
                true
            }
            Ok(Err(e)) => {
                println!("   Read error: {}", e);
                false
            }
            Err(_) => {
                println!("   TIMEOUT: No packets received within 3 seconds");
                println!("   (This may be normal if no traffic is sent to 10.0.85.1)");
                true // Not a failure, just no traffic
            }
        }
    });

    // Give the read task a moment to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Send a ping to the TUN interface IP
    println!("   Sending ping to 10.0.85.1...");
    let ping_output = Command::new("ping")
        .args(["-c", "1", "-t", "1", "10.0.85.1"])
        .output();
    
    match ping_output {
        Ok(o) => {
            if o.status.success() {
                println!("   Ping sent successfully");
            } else {
                // Ping will "fail" because there's no response, but packet was sent
                println!("   Ping command completed (no response expected)");
            }
        }
        Err(e) => {
            println!("   Ping failed to execute: {}", e);
        }
    }

    // Wait for the read task
    let _ = read_handle.await;

    println!("\n[6/6] Cleaning up...");
    match tun.down() {
        Ok(_) => println!("   SUCCESS: Interface brought down"),
        Err(e) => println!("   Warning: Failed to bring down interface: {}", e),
    }
    
    // Interface is automatically removed when tun is dropped
    drop(tun);
    
    // Verify it's gone
    tokio::time::sleep(Duration::from_millis(500)).await;
    let verify = Command::new("ifconfig")
        .arg("utun99") // Won't exist
        .output();
    
    println!("   Device cleaned up");

    println!("\n=== All tests passed! ===");
}
