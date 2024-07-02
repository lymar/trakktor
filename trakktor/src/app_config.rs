pub trait AppConfigProvider {
    fn is_dev_mode(&self) -> bool;
}
