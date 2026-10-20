/// Choices for creating or opening a database.
///
/// The device name is the writer identity stored in the `.RR` claim. It is not
/// an account, and it is not read from the environment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenOptions {
    pub device_name: String,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            device_name: "local".to_string(),
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
}
