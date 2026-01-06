//! Signal handling for graceful shutdown and configuration reload.
//!
//! Handles:
//! - SIGTERM/SIGINT for graceful shutdown
//! - SIGHUP for configuration reload
//! - SIGUSR1/SIGUSR2 for custom actions

use std::sync::Arc;

use parking_lot::RwLock;
use tokio::sync::broadcast;
use tracing::{debug, error, info};

/// Signal type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// Terminate signal (SIGTERM).
    Terminate,
    /// Interrupt signal (SIGINT).
    Interrupt,
    /// Hangup signal (SIGHUP) - typically reload configuration.
    Hangup,
    /// User signal 1 (SIGUSR1).
    User1,
    /// User signal 2 (SIGUSR2).
    User2,
    /// Child signal (SIGCHLD).
    Child,
}

impl std::fmt::Display for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Signal::Terminate => write!(f, "SIGTERM"),
            Signal::Interrupt => write!(f, "SIGINT"),
            Signal::Hangup => write!(f, "SIGHUP"),
            Signal::User1 => write!(f, "SIGUSR1"),
            Signal::User2 => write!(f, "SIGUSR2"),
            Signal::Child => write!(f, "SIGCHLD"),
        }
    }
}

/// Signal handler that broadcasts signals to subscribers.
pub struct SignalHandler {
    /// Shutdown flag.
    shutdown: Arc<RwLock<bool>>,
    /// Signal broadcaster.
    signal_tx: broadcast::Sender<Signal>,
    /// Reload callback.
    reload_callback: Arc<RwLock<Option<Box<dyn Fn() + Send + Sync>>>>,
}

impl SignalHandler {
    /// Create a new signal handler.
    pub fn new() -> Self {
        let (signal_tx, _) = broadcast::channel(16);

        Self {
            shutdown: Arc::new(RwLock::new(false)),
            signal_tx,
            reload_callback: Arc::new(RwLock::new(None)),
        }
    }

    /// Subscribe to signals.
    pub fn subscribe(&self) -> broadcast::Receiver<Signal> {
        self.signal_tx.subscribe()
    }

    /// Check if shutdown was requested.
    pub fn is_shutdown(&self) -> bool {
        *self.shutdown.read()
    }

    /// Request shutdown.
    pub fn request_shutdown(&self) {
        *self.shutdown.write() = true;
        let _ = self.signal_tx.send(Signal::Terminate);
    }

    /// Set the reload callback.
    pub fn set_reload_callback<F>(&self, callback: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        *self.reload_callback.write() = Some(Box::new(callback));
    }

    /// Handle a signal.
    fn handle_signal(&self, signal: Signal) {
        info!("Received signal: {}", signal);

        match signal {
            Signal::Terminate | Signal::Interrupt => {
                info!("Initiating graceful shutdown");
                self.request_shutdown();
            }
            Signal::Hangup => {
                info!("Reloading configuration");
                if let Some(callback) = self.reload_callback.read().as_ref() {
                    callback();
                }
            }
            Signal::User1 => {
                // Custom action - could be used for log rotation, stats dump, etc.
                debug!("SIGUSR1 received");
            }
            Signal::User2 => {
                // Custom action
                debug!("SIGUSR2 received");
            }
            Signal::Child => {
                // Child process exited
                debug!("SIGCHLD received");
            }
        }

        let _ = self.signal_tx.send(signal);
    }

    /// Start listening for signals (Unix).
    #[cfg(unix)]
    pub async fn listen(&self) {
        use futures::StreamExt;
        use signal_hook::consts::signal::*;
        use signal_hook_tokio::Signals;

        let signals = match Signals::new(&[SIGTERM, SIGINT, SIGHUP, SIGUSR1, SIGUSR2]) {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to register signal handlers: {}", e);
                return;
            }
        };

        let mut signals = signals.fuse();

        info!("Signal handler started");

        while let Some(signal) = signals.next().await {
            let sig = match signal {
                SIGTERM => Signal::Terminate,
                SIGINT => Signal::Interrupt,
                SIGHUP => Signal::Hangup,
                SIGUSR1 => Signal::User1,
                SIGUSR2 => Signal::User2,
                SIGCHLD => Signal::Child,
                _ => continue,
            };

            self.handle_signal(sig);

            // Exit loop on termination signals
            if sig == Signal::Terminate || sig == Signal::Interrupt {
                break;
            }
        }

        info!("Signal handler stopped");
    }

    /// Start listening for signals (non-Unix fallback).
    #[cfg(not(unix))]
    pub async fn listen(&self) {
        // On non-Unix platforms, just wait for Ctrl+C
        match tokio::signal::ctrl_c().await {
            Ok(()) => {
                self.handle_signal(Signal::Interrupt);
            }
            Err(e) => {
                error!("Failed to listen for Ctrl+C: {}", e);
            }
        }
    }

    /// Create a shutdown future that completes when shutdown is requested.
    pub fn shutdown_signal(&self) -> ShutdownSignal {
        ShutdownSignal {
            shutdown: Arc::clone(&self.shutdown),
            rx: self.signal_tx.subscribe(),
        }
    }
}

impl Default for SignalHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// A future that completes when shutdown is signaled.
pub struct ShutdownSignal {
    shutdown: Arc<RwLock<bool>>,
    rx: broadcast::Receiver<Signal>,
}

impl ShutdownSignal {
    /// Wait for shutdown.
    pub async fn wait(&mut self) {
        if *self.shutdown.read() {
            return;
        }

        loop {
            match self.rx.recv().await {
                Ok(Signal::Terminate) | Ok(Signal::Interrupt) => {
                    return;
                }
                Ok(_) => continue,
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => return,
            }
        }
    }
}

impl std::future::Future for ShutdownSignal {
    type Output = ();

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if *self.shutdown.read() {
            return std::task::Poll::Ready(());
        }

        // Try to receive without blocking
        match self.rx.try_recv() {
            Ok(Signal::Terminate) | Ok(Signal::Interrupt) => std::task::Poll::Ready(()),
            Ok(_) => {
                cx.waker().wake_by_ref();
                std::task::Poll::Pending
            }
            Err(broadcast::error::TryRecvError::Empty) => {
                cx.waker().wake_by_ref();
                std::task::Poll::Pending
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => {
                cx.waker().wake_by_ref();
                std::task::Poll::Pending
            }
            Err(broadcast::error::TryRecvError::Closed) => std::task::Poll::Ready(()),
        }
    }
}

/// Convenience function to wait for shutdown signals.
pub async fn wait_for_shutdown() {
    let handler = SignalHandler::new();
    handler.listen().await;
}

/// Setup signal handlers and return a shutdown receiver.
pub fn setup_signal_handlers() -> (Arc<SignalHandler>, broadcast::Receiver<Signal>) {
    let handler = Arc::new(SignalHandler::new());
    let rx = handler.subscribe();

    // Start listening in background
    let handler_clone = Arc::clone(&handler);
    tokio::spawn(async move {
        handler_clone.listen().await;
    });

    (handler, rx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_display() {
        assert_eq!(format!("{}", Signal::Terminate), "SIGTERM");
        assert_eq!(format!("{}", Signal::Interrupt), "SIGINT");
        assert_eq!(format!("{}", Signal::Hangup), "SIGHUP");
    }

    #[test]
    fn test_signal_handler_shutdown() {
        let handler = SignalHandler::new();

        assert!(!handler.is_shutdown());
        handler.request_shutdown();
        assert!(handler.is_shutdown());
    }

    #[test]
    fn test_reload_callback() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let handler = SignalHandler::new();
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = Arc::clone(&called);

        handler.set_reload_callback(move || {
            called_clone.store(true, Ordering::SeqCst);
        });

        handler.handle_signal(Signal::Hangup);

        assert!(called.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_shutdown_signal() {
        let handler = SignalHandler::new();
        let mut shutdown = handler.shutdown_signal();

        // Request shutdown in background
        let handler_clone = handler.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            handler_clone.request_shutdown();
        });

        // Wait for shutdown
        tokio::time::timeout(std::time::Duration::from_millis(100), shutdown.wait())
            .await
            .expect("Shutdown should complete");
    }
}

impl Clone for SignalHandler {
    fn clone(&self) -> Self {
        Self {
            shutdown: Arc::clone(&self.shutdown),
            signal_tx: self.signal_tx.clone(),
            reload_callback: Arc::clone(&self.reload_callback),
        }
    }
}
