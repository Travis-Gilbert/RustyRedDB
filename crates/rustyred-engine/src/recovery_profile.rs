use serde::{Deserialize, Serialize};
#[cfg(target_os = "linux")]
use std::fs;

/// One bounded measurement captured while a durable graph becomes resident.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecoveryStageSample {
    pub stage: String,
    pub index_kind: Option<String>,
    pub journal_bytes: u64,
    pub live_nodes: usize,
    pub live_edges: usize,
    pub rss_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
}

/// Recovery samples are retained on the opened store so both boot logs and the
/// offline volume oracle report the exact same observations.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecoveryProfile {
    pub samples: Vec<RecoveryStageSample>,
}

impl RecoveryProfile {
    pub(crate) fn capture(
        &mut self,
        stage: impl Into<String>,
        index_kind: Option<impl Into<String>>,
        journal_bytes: u64,
        live_nodes: usize,
        live_edges: usize,
    ) {
        let (rss_bytes, peak_rss_bytes) = process_memory_bytes();
        self.samples.push(RecoveryStageSample {
            stage: stage.into(),
            index_kind: index_kind.map(Into::into),
            journal_bytes,
            live_nodes,
            live_edges,
            rss_bytes,
            peak_rss_bytes,
        });
    }

    pub fn peak_rss_bytes(&self) -> Option<u64> {
        self.samples
            .iter()
            .filter_map(|sample| sample.peak_rss_bytes.or(sample.rss_bytes))
            .max()
    }
}

#[cfg(target_os = "linux")]
fn process_memory_bytes() -> (Option<u64>, Option<u64>) {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return (None, None);
    };
    (
        status_kib(&status, "VmRSS").map(kib_to_bytes),
        status_kib(&status, "VmHWM").map(kib_to_bytes),
    )
}

#[cfg(target_os = "macos")]
fn process_memory_bytes() -> (Option<u64>, Option<u64>) {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage initializes the provided rusage on success, and the
    // pointer is valid for the duration of the call.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return (None, None);
    }
    // macOS reports ru_maxrss in bytes (unlike Linux's KiB /proc fields).
    let peak = unsafe { usage.assume_init() }.ru_maxrss;
    let peak = u64::try_from(peak).ok();
    (None, peak)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn process_memory_bytes() -> (Option<u64>, Option<u64>) {
    (None, None)
}

#[cfg(target_os = "linux")]
fn status_kib(status: &str, key: &str) -> Option<u64> {
    status.lines().find_map(|line| {
        let value = line.strip_prefix(key)?.trim_start_matches(':').trim();
        value.split_whitespace().next()?.parse::<u64>().ok()
    })
}

#[cfg(target_os = "linux")]
fn kib_to_bytes(kib: u64) -> u64 {
    kib.saturating_mul(1024)
}

#[cfg(test)]
mod tests {
    use super::RecoveryProfile;

    #[test]
    fn profile_retains_named_stages_and_peak() {
        let mut profile = RecoveryProfile::default();
        profile.capture("journal_open", None::<String>, 42, 0, 0);
        profile.capture("live_state", Some("node_indexes"), 42, 3, 0);
        assert_eq!(profile.samples[0].stage, "journal_open");
        assert_eq!(
            profile.samples[1].index_kind.as_deref(),
            Some("node_indexes")
        );
        assert_eq!(profile.samples[1].live_nodes, 3);
    }
}
