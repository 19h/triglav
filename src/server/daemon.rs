//! Daemon mode implementation.
//!
//! Provides proper Unix daemon functionality including:
//! - Background forking
//! - PID file management
//! - Privilege dropping
//! - Signal handling

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;

use tracing::{info, warn};

use crate::error::{Error, Result};

/// Daemon configuration.
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// Path to PID file.
    pub pid_file: Option<PathBuf>,
    /// Working directory.
    pub work_dir: PathBuf,
    /// User to run as (name or UID).
    pub user: Option<String>,
    /// Group to run as (name or GID).
    pub group: Option<String>,
    /// Umask for created files.
    pub umask: Option<u32>,
    /// Close standard file descriptors.
    pub close_fds: bool,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            pid_file: None,
            work_dir: PathBuf::from("/"),
            user: None,
            group: None,
            umask: Some(0o027),
            close_fds: true,
        }
    }
}

impl DaemonConfig {
    /// Set the PID file path.
    pub fn with_pid_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.pid_file = Some(path.into());
        self
    }

    /// Set the working directory.
    pub fn with_work_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.work_dir = path.into();
        self
    }

    /// Set the user to run as.
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Set the group to run as.
    pub fn with_group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }
}

/// Daemonize the current process.
#[cfg(unix)]
pub fn daemonize(config: &DaemonConfig) -> Result<()> {
    use nix::sys::stat;
    use nix::unistd::{chdir, fork, setsid, ForkResult};
    use std::os::unix::io::AsRawFd;

    // Check for existing PID file
    if let Some(ref pid_file) = config.pid_file {
        if pid_file.exists() {
            // Check if process is still running
            if let Ok(content) = fs::read_to_string(pid_file) {
                if let Ok(pid) = content.trim().parse::<i32>() {
                    if is_process_running(pid) {
                        return Err(Error::Config(format!(
                            "Daemon already running with PID {} (from {})",
                            pid,
                            pid_file.display()
                        )));
                    }
                }
            }
            // Stale PID file, remove it
            let _ = fs::remove_file(pid_file);
        }
    }

    // First fork
    match unsafe { fork() } {
        Ok(ForkResult::Parent { .. }) => {
            // Parent exits
            process::exit(0);
        }
        Ok(ForkResult::Child) => {
            // Continue as child
        }
        Err(e) => {
            return Err(Error::Config(format!("First fork failed: {}", e)));
        }
    }

    // Create new session
    setsid().map_err(|e| Error::Config(format!("setsid failed: {}", e)))?;

    // Second fork (prevent acquiring controlling terminal)
    match unsafe { fork() } {
        Ok(ForkResult::Parent { .. }) => {
            process::exit(0);
        }
        Ok(ForkResult::Child) => {
            // Continue as grandchild
        }
        Err(e) => {
            return Err(Error::Config(format!("Second fork failed: {}", e)));
        }
    }

    // Set umask
    if let Some(mask) = config.umask {
        stat::umask(stat::Mode::from_bits_truncate(mask as libc::mode_t));
    }

    // Change working directory
    chdir(&config.work_dir).map_err(|e| Error::Config(format!("chdir failed: {}", e)))?;

    // Close standard file descriptors and redirect to /dev/null
    if config.close_fds {
        let devnull = File::open("/dev/null")
            .map_err(|e| Error::Config(format!("Failed to open /dev/null: {}", e)))?;
        let fd = devnull.as_raw_fd();

        unsafe {
            libc::dup2(fd, 0); // stdin
            libc::dup2(fd, 1); // stdout
            libc::dup2(fd, 2); // stderr
        }
    }

    // Drop privileges if configured
    if config.user.is_some() || config.group.is_some() {
        drop_privileges(&config.user, &config.group)?;
    }

    // Write PID file
    if let Some(ref pid_file) = config.pid_file {
        write_pid_file(pid_file)?;
    }

    info!("Daemonized successfully (PID: {})", process::id());

    Ok(())
}

#[cfg(not(unix))]
pub fn daemonize(_config: &DaemonConfig) -> Result<()> {
    Err(Error::Config(
        "Daemon mode is not supported on this platform".into(),
    ))
}

/// Write the current PID to a file.
pub fn write_pid_file(path: &Path) -> Result<()> {
    let pid = process::id();

    // Create parent directories if needed
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| Error::Config(format!("Failed to create PID file directory: {}", e)))?;
    }

    let mut file = File::create(path)
        .map_err(|e| Error::Config(format!("Failed to create PID file: {}", e)))?;

    writeln!(file, "{}", pid)
        .map_err(|e| Error::Config(format!("Failed to write PID file: {}", e)))?;

    info!("Wrote PID {} to {}", pid, path.display());

    Ok(())
}

/// Remove the PID file.
pub fn remove_pid_file(path: &Path) -> Result<()> {
    if path.exists() {
        fs::remove_file(path)
            .map_err(|e| Error::Config(format!("Failed to remove PID file: {}", e)))?;
        debug!("Removed PID file {}", path.display());
    }
    Ok(())
}

/// Check if a process is running.
#[cfg(unix)]
fn is_process_running(pid: i32) -> bool {
    // Send signal 0 to check if process exists
    unsafe { libc::kill(pid, 0) == 0 }
}

#[cfg(not(unix))]
fn is_process_running(_pid: i32) -> bool {
    false
}

/// Drop privileges to a specific user/group.
#[cfg(unix)]
fn drop_privileges(user: &Option<String>, group: &Option<String>) -> Result<()> {
    use nix::unistd::{setgid, setuid, Gid, Uid};

    // Set group first (can't change group after dropping user privileges)
    if let Some(group_name) = group {
        let gid = resolve_group(group_name)?;
        setgid(Gid::from_raw(gid)).map_err(|e| Error::Config(format!("setgid failed: {}", e)))?;
        info!("Set group to {} (gid={})", group_name, gid);
    }

    // Set user
    if let Some(user_name) = user {
        let uid = resolve_user(user_name)?;
        setuid(Uid::from_raw(uid)).map_err(|e| Error::Config(format!("setuid failed: {}", e)))?;
        info!("Set user to {} (uid={})", user_name, uid);
    }

    Ok(())
}

/// Resolve a username or UID to a numeric UID.
#[cfg(unix)]
fn resolve_user(user: &str) -> Result<u32> {
    use std::ffi::CString;

    // Try parsing as numeric UID first
    if let Ok(uid) = user.parse::<u32>() {
        return Ok(uid);
    }

    // Look up by name
    let cname = CString::new(user).map_err(|_| Error::Config("Invalid user name".into()))?;

    let pwd = unsafe { libc::getpwnam(cname.as_ptr()) };
    if pwd.is_null() {
        return Err(Error::Config(format!("User not found: {}", user)));
    }

    Ok(unsafe { (*pwd).pw_uid })
}

/// Resolve a group name or GID to a numeric GID.
#[cfg(unix)]
fn resolve_group(group: &str) -> Result<u32> {
    use std::ffi::CString;

    // Try parsing as numeric GID first
    if let Ok(gid) = group.parse::<u32>() {
        return Ok(gid);
    }

    // Look up by name
    let cname = CString::new(group).map_err(|_| Error::Config("Invalid group name".into()))?;

    let grp = unsafe { libc::getgrnam(cname.as_ptr()) };
    if grp.is_null() {
        return Err(Error::Config(format!("Group not found: {}", group)));
    }

    Ok(unsafe { (*grp).gr_gid })
}

/// PID file guard that removes the file on drop.
pub struct PidFileGuard {
    path: PathBuf,
}

impl PidFileGuard {
    /// Create a new PID file guard.
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        write_pid_file(&path)?;
        Ok(Self { path })
    }
}

impl Drop for PidFileGuard {
    fn drop(&mut self) {
        if let Err(e) = remove_pid_file(&self.path) {
            warn!("Failed to remove PID file: {}", e);
        }
    }
}

use tracing::debug;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_pid_file() {
        let dir = tempdir().unwrap();
        let pid_path = dir.path().join("test.pid");

        write_pid_file(&pid_path).unwrap();

        let content = fs::read_to_string(&pid_path).unwrap();
        let pid: u32 = content.trim().parse().unwrap();
        assert_eq!(pid, process::id());

        remove_pid_file(&pid_path).unwrap();
        assert!(!pid_path.exists());
    }

    #[test]
    fn test_pid_file_guard() {
        let dir = tempdir().unwrap();
        let pid_path = dir.path().join("guard.pid");

        {
            let _guard = PidFileGuard::new(&pid_path).unwrap();
            assert!(pid_path.exists());
        }

        // Should be removed on drop
        assert!(!pid_path.exists());
    }

    #[test]
    fn test_daemon_config_builder() {
        let config = DaemonConfig::default()
            .with_pid_file("/var/run/test.pid")
            .with_work_dir("/tmp")
            .with_user("nobody")
            .with_group("nogroup");

        assert_eq!(config.pid_file, Some(PathBuf::from("/var/run/test.pid")));
        assert_eq!(config.work_dir, PathBuf::from("/tmp"));
        assert_eq!(config.user, Some("nobody".to_string()));
        assert_eq!(config.group, Some("nogroup".to_string()));
    }

    #[cfg(unix)]
    #[test]
    fn test_is_process_running() {
        // Current process should be running
        assert!(is_process_running(process::id() as i32));

        // Invalid PID should not be running
        assert!(!is_process_running(999999999));
    }
}
