/// Choices for creating or opening a database.
///
/// The device name is the writer identity stored in the `.RR` claim. It is not
/// an account, and it is not read from the environment.
///
/// The other two choices are for a database that holds derived data, such as
/// an index rebuilt from files. Both are on by default, belong to this handle
/// and are not stored in the file.
///
/// - `control_events`: a write to a labelled node also commits a
///   `ControlEvent` node recording it. Off, a write adds only the records it
///   names.
/// - `keep_history`: each commit is added to the recorded revision history
///   and to the change log `changes_since` reads. Off, the file keeps only its
///   current state, so [`crate::Database::compact`] can reclaim what each
///   write replaced and the file stays near the size of its live data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenOptions {
    pub device_name: String,
    pub control_events: bool,
    pub keep_history: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            device_name: "local".to_string(),
            control_events: true,
            keep_history: true,
        }
    }
}

impl OpenOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn device_name(mut self, name: impl Into<String>) -> Self {
        self.device_name = name.into();
        self
    }

    pub fn control_events(mut self, enabled: bool) -> Self {
        self.control_events = enabled;
        self
    }

    pub fn keep_history(mut self, keep: bool) -> Self {
        self.keep_history = keep;
        self
    }
}
