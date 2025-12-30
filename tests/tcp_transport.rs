//! TCP transport tests - validates TCP fallback functionality.
//!
//! These tests ensure TCP transport works correctly for networks where UDP is blocked.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::broadcast;

use triglav::transport::{TcpTransport, Transport, TransportConfig};
use triglav::error::Result;

// ============================================================================
// Basic TCP Transport Tests
// ============================================================================

#[tokio::test]
async fn test_tcp_transport_bind() {
    let config = TransportConfig::default();
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    
    let transport = TcpTransport::bind(addr, &config);
    assert!(transport.is_ok(), "Should be able to bind TCP transport");
    
    let transport = transport.unwrap();
    let local_addr = transport.local_addr().unwrap();
    assert!(local_addr.port() > 0, "Should have a valid port");
    assert_eq!(transport.transport_type(), "tcp");
}

#[tokio::test]
async fn test_tcp_transport_connect() {
    // Start a simple TCP server
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap();
    
    // Accept connections in background
    let (shutdown_tx, _) = broadcast::channel::<()>(1);
    let mut shutdown_rx = shutdown_tx.subscribe();
    tokio::spawn(async move {
        tokio::select! {
            result = listener.accept() => {
                if let Ok((mut stream, _)) = result {
                    let mut buf = [0u8; 1024];
                    if let Ok(n) = stream.read(&mut buf).await {
                        // Echo back
                        let _ = stream.write_all(&buf[..n]).await;
                    }
                }
            }
            _ = shutdown_rx.recv() => {}
        }
    });
    
    tokio::time::sleep(Duration::from_millis(50)).await;
    
    // Connect using TcpTransport
    let config = TransportConfig::default();
    let transport = TcpTransport::connect(server_addr, None, &config).await;
    
    assert!(transport.is_ok(), "Should be able to connect: {:?}", transport.err());
    let transport = transport.unwrap();
    
    assert!(transport.is_connected(), "Should be connected");
    assert_eq!(transport.remote_addr(), Some(server_addr));
    
    let _ = shutdown_tx.send(());
}

#[tokio::test]
async fn test_tcp_transport_send_recv() {
    // Start echo server
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap();
    
    tokio::spawn(async move {
        if let Ok((mut stream, _)) = listener.accept().await {
            let mut buf = [0u8; 4096];
            loop {
                // Read length prefix
                let mut len_buf = [0u8; 4];
                if stream.read_exact(&mut len_buf).await.is_err() {
                    break;
                }
                let len = u32::from_be_bytes(len_buf) as usize;
                
                // Read data
                if stream.read_exact(&mut buf[..len]).await.is_err() {
                    break;
                }
                
                // Echo back with length prefix
                if stream.write_all(&len_buf).await.is_err() {
                    break;
                }
                if stream.write_all(&buf[..len]).await.is_err() {
                    break;
                }
            }
        }
    });
    
    tokio::time::sleep(Duration::from_millis(50)).await;
    
    let config = TransportConfig::default();
    let transport = TcpTransport::connect(server_addr, None, &config).await.unwrap();
    
    // Send test data
    let test_data = b"Hello, TCP Transport!";
    let sent = transport.send(test_data).await.unwrap();
    assert_eq!(sent, test_data.len());
    
    // Receive echo
    let mut buf = [0u8; 1024];
    let received = transport.recv(&mut buf).await.unwrap();
    assert_eq!(received, test_data.len());
    assert_eq!(&buf[..received], test_data);
}

#[tokio::test]
async fn test_tcp_transport_large_payload() {
    // Start echo server
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap();
    
    tokio::spawn(async move {
        if let Ok((mut stream, _)) = listener.accept().await {
            let mut buf = vec![0u8; 65536];
            loop {
                // Read length prefix
                let mut len_buf = [0u8; 4];
                if stream.read_exact(&mut len_buf).await.is_err() {
                    break;
                }
                let len = u32::from_be_bytes(len_buf) as usize;
                
                // Read data
                if stream.read_exact(&mut buf[..len]).await.is_err() {
                    break;
                }
                
                // Echo back
                if stream.write_all(&len_buf).await.is_err() {
                    break;
                }
                if stream.write_all(&buf[..len]).await.is_err() {
                    break;
                }
            }
        }
    });
    
    tokio::time::sleep(Duration::from_millis(50)).await;
    
    let config = TransportConfig::default();
    let transport = TcpTransport::connect(server_addr, None, &config).await.unwrap();
    
    // Send large payload (10KB)
    let large_data: Vec<u8> = (0..10240).map(|i| (i % 256) as u8).collect();
    let sent = transport.send(&large_data).await.unwrap();
    assert_eq!(sent, large_data.len());
    
    // Receive echo
    let mut buf = vec![0u8; 65536];
    let received = transport.recv(&mut buf).await.unwrap();
    assert_eq!(received, large_data.len());
    assert_eq!(&buf[..received], &large_data[..]);
}

#[tokio::test]
async fn test_tcp_transport_multiple_messages() {
    // Start echo server
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap();
    
    tokio::spawn(async move {
        if let Ok((mut stream, _)) = listener.accept().await {
            let mut buf = vec![0u8; 4096];
            loop {
                let mut len_buf = [0u8; 4];
                if stream.read_exact(&mut len_buf).await.is_err() {
                    break;
                }
                let len = u32::from_be_bytes(len_buf) as usize;
                if stream.read_exact(&mut buf[..len]).await.is_err() {
                    break;
                }
                if stream.write_all(&len_buf).await.is_err() {
                    break;
                }
                if stream.write_all(&buf[..len]).await.is_err() {
                    break;
                }
            }
        }
    });
    
    tokio::time::sleep(Duration::from_millis(50)).await;
    
    let config = TransportConfig::default();
    let transport = TcpTransport::connect(server_addr, None, &config).await.unwrap();
    
    // Send multiple messages
    for i in 0..10 {
        let msg = format!("Message number {}", i);
        transport.send(msg.as_bytes()).await.unwrap();
        
        let mut buf = [0u8; 1024];
        let n = transport.recv(&mut buf).await.unwrap();
        assert_eq!(&buf[..n], msg.as_bytes());
    }
}

#[tokio::test]
async fn test_tcp_transport_close() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap();
    
    tokio::spawn(async move {
        let _ = listener.accept().await;
    });
    
    tokio::time::sleep(Duration::from_millis(50)).await;
    
    let config = TransportConfig::default();
    let transport = TcpTransport::connect(server_addr, None, &config).await.unwrap();
    
    assert!(transport.is_connected());
    
    transport.close().await.unwrap();
    
    // After close, is_connected should return false
    assert!(!transport.is_connected());
}

#[tokio::test]
async fn test_tcp_transport_connection_refused() {
    let config = TransportConfig {
        connect_timeout: Duration::from_millis(500),
        ..Default::default()
    };
    
    // Try to connect to a port that's not listening
    let result = TcpTransport::connect("127.0.0.1:59999".parse().unwrap(), None, &config).await;
    
    assert!(result.is_err(), "Should fail to connect to non-listening port");
}

#[tokio::test]
async fn test_tcp_transport_with_bind_addr() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap();
    
    tokio::spawn(async move {
        let _ = listener.accept().await;
    });
    
    tokio::time::sleep(Duration::from_millis(50)).await;
    
    let config = TransportConfig::default();
    let bind_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    
    let transport = TcpTransport::connect(server_addr, Some(bind_addr), &config).await.unwrap();
    
    let local = transport.local_addr().unwrap();
    assert_eq!(local.ip(), bind_addr.ip());
}

// ============================================================================
// TCP Transport Accept Tests (Server Mode)
// ============================================================================

#[tokio::test]
async fn test_tcp_transport_accept() {
    let config = TransportConfig::default();
    let server = TcpTransport::bind("127.0.0.1:0".parse().unwrap(), &config).unwrap();
    let server_addr = server.local_addr().unwrap();
    
    // Connect from a client
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let _ = tokio::net::TcpStream::connect(server_addr).await;
    });
    
    // Accept connection
    let result = tokio::time::timeout(Duration::from_secs(2), server.accept()).await;
    
    assert!(result.is_ok(), "Accept should not timeout");
    let (stream, addr) = result.unwrap().unwrap();
    assert!(addr.port() > 0);
    
    // Verify we can get addresses from the stream
    assert!(stream.local_addr().is_ok());
    assert!(stream.peer_addr().is_ok());
}

// ============================================================================
// TCP Transport TcpStream Wrapper Tests
// ============================================================================

#[tokio::test]
async fn test_tcp_stream_wrapper_send_recv() {
    use triglav::transport::TcpStream;
    
    // Create connected pair
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap();
    
    let client_task = tokio::spawn(async move {
        tokio::net::TcpStream::connect(server_addr).await.unwrap()
    });
    
    let (server_stream, _) = listener.accept().await.unwrap();
    let client_stream = client_task.await.unwrap();
    
    let mut server = TcpStream::new(server_stream);
    let mut client = TcpStream::new(client_stream);
    
    // Client sends, server receives
    let test_data = b"Test message via TcpStream wrapper";
    client.send(test_data).await.unwrap();
    
    let mut buf = [0u8; 1024];
    let n = server.recv(&mut buf).await.unwrap();
    assert_eq!(&buf[..n], test_data);
    
    // Server sends, client receives
    let response = b"Response from server";
    server.send(response).await.unwrap();
    
    let n = client.recv(&mut buf).await.unwrap();
    assert_eq!(&buf[..n], response);
}

#[tokio::test]
async fn test_tcp_stream_shutdown() {
    use triglav::transport::TcpStream;
    
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap();
    
    let client_task = tokio::spawn(async move {
        tokio::net::TcpStream::connect(server_addr).await.unwrap()
    });
    
    let (server_stream, _) = listener.accept().await.unwrap();
    let client_stream = client_task.await.unwrap();
    
    let mut server = TcpStream::new(server_stream);
    let mut client = TcpStream::new(client_stream);
    
    // Shutdown client
    client.shutdown().await.unwrap();
    
    // Server should detect connection closed
    let mut buf = [0u8; 4];
    let result = server.recv(&mut buf).await;
    assert!(result.is_err() || result.unwrap() == 0);
}

// ============================================================================
// TCP Transport Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_tcp_send_on_closed_connection() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap();
    
    // Accept and immediately close
    tokio::spawn(async move {
        if let Ok((stream, _)) = listener.accept().await {
            drop(stream); // Close immediately
        }
    });
    
    tokio::time::sleep(Duration::from_millis(50)).await;
    
    let config = TransportConfig::default();
    let transport = TcpTransport::connect(server_addr, None, &config).await.unwrap();
    
    // Wait for server to close
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Close our end
    transport.close().await.unwrap();
    
    // Now send should fail
    let result = transport.send(b"test").await;
    assert!(result.is_err(), "Send on closed connection should fail");
}

#[tokio::test]
async fn test_tcp_recv_message_too_large() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap();
    
    // Server sends a message claiming to be very large
    tokio::spawn(async move {
        if let Ok((mut stream, _)) = listener.accept().await {
            // Send length prefix claiming 100KB
            let len: u32 = 100_000;
            stream.write_all(&len.to_be_bytes()).await.unwrap();
            // Don't actually send that much data
        }
    });
    
    tokio::time::sleep(Duration::from_millis(50)).await;
    
    let config = TransportConfig::default();
    let transport = TcpTransport::connect(server_addr, None, &config).await.unwrap();
    
    // Try to receive into a small buffer
    let mut buf = [0u8; 1024];
    let result = transport.recv(&mut buf).await;
    
    assert!(result.is_err(), "Should fail when message is larger than buffer");
}

// ============================================================================
// TCP Transport Configuration Tests
// ============================================================================

#[tokio::test]
async fn test_tcp_transport_nodelay() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap();
    
    tokio::spawn(async move {
        let _ = listener.accept().await;
    });
    
    tokio::time::sleep(Duration::from_millis(50)).await;
    
    let config = TransportConfig {
        tcp_nodelay: true,
        ..Default::default()
    };
    
    let transport = TcpTransport::connect(server_addr, None, &config).await;
    assert!(transport.is_ok(), "Should connect with TCP_NODELAY enabled");
}

#[tokio::test]
async fn test_tcp_transport_custom_timeout() {
    let config = TransportConfig {
        connect_timeout: Duration::from_millis(100),
        ..Default::default()
    };
    
    // Try to connect to a non-routable address (should timeout)
    let start = std::time::Instant::now();
    let result = TcpTransport::connect("10.255.255.1:80".parse().unwrap(), None, &config).await;
    let elapsed = start.elapsed();
    
    assert!(result.is_err(), "Should fail to connect");
    // Should timeout around the configured duration (with some tolerance)
    assert!(elapsed < Duration::from_secs(5), "Should timeout quickly");
}

// ============================================================================
// TCP Transport IPv6 Tests
// ============================================================================

#[tokio::test]
async fn test_tcp_transport_ipv6() {
    // Try to bind to IPv6 loopback
    let listener = match TcpListener::bind("[::1]:0").await {
        Ok(l) => l,
        Err(_) => {
            // IPv6 might not be available
            println!("IPv6 not available, skipping test");
            return;
        }
    };
    let server_addr = listener.local_addr().unwrap();
    
    tokio::spawn(async move {
        if let Ok((mut stream, _)) = listener.accept().await {
            let mut buf = vec![0u8; 4096];
            loop {
                let mut len_buf = [0u8; 4];
                if stream.read_exact(&mut len_buf).await.is_err() {
                    break;
                }
                let len = u32::from_be_bytes(len_buf) as usize;
                if stream.read_exact(&mut buf[..len]).await.is_err() {
                    break;
                }
                if stream.write_all(&len_buf).await.is_err() {
                    break;
                }
                if stream.write_all(&buf[..len]).await.is_err() {
                    break;
                }
            }
        }
    });
    
    tokio::time::sleep(Duration::from_millis(50)).await;
    
    let config = TransportConfig::default();
    let transport = TcpTransport::connect(server_addr, None, &config).await;
    
    assert!(transport.is_ok(), "Should connect over IPv6: {:?}", transport.err());
    
    let transport = transport.unwrap();
    let test_data = b"IPv6 test message";
    transport.send(test_data).await.unwrap();
    
    let mut buf = [0u8; 1024];
    let n = transport.recv(&mut buf).await.unwrap();
    assert_eq!(&buf[..n], test_data);
}

// ============================================================================
// TCP Transport Concurrent Connections Test
// ============================================================================

#[tokio::test]
async fn test_tcp_transport_concurrent_connections() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap();
    
    // Server handles multiple connections
    tokio::spawn(async move {
        loop {
            if let Ok((mut stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 4096];
                    loop {
                        let mut len_buf = [0u8; 4];
                        if stream.read_exact(&mut len_buf).await.is_err() {
                            break;
                        }
                        let len = u32::from_be_bytes(len_buf) as usize;
                        if stream.read_exact(&mut buf[..len]).await.is_err() {
                            break;
                        }
                        if stream.write_all(&len_buf).await.is_err() {
                            break;
                        }
                        if stream.write_all(&buf[..len]).await.is_err() {
                            break;
                        }
                    }
                });
            }
        }
    });
    
    tokio::time::sleep(Duration::from_millis(50)).await;
    
    let config = TransportConfig::default();
    
    // Create multiple concurrent connections
    let mut handles = vec![];
    for i in 0..10 {
        let server_addr = server_addr;
        let config = config.clone();
        handles.push(tokio::spawn(async move {
            let transport = TcpTransport::connect(server_addr, None, &config).await.unwrap();
            
            let msg = format!("Connection {} message", i);
            transport.send(msg.as_bytes()).await.unwrap();
            
            let mut buf = [0u8; 1024];
            let n = transport.recv(&mut buf).await.unwrap();
            assert_eq!(&buf[..n], msg.as_bytes());
            
            i
        }));
    }
    
    // Wait for all to complete
    let mut completed = 0;
    for handle in handles {
        let _ = handle.await.unwrap();
        completed += 1;
    }
    
    assert_eq!(completed, 10, "All concurrent connections should complete");
}
