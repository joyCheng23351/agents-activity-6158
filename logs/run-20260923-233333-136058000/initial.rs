//! Recreated starter for an isolated live run; same API and six unimplemented functions.
use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub prerelease: Option<String>,
    pub build: Option<String>,
}

pub fn parse(_s: &str) -> Result<Version, String> {
    todo!("parse")
}

pub fn to_string(_v: &Version) -> String {
    todo!("to_string")
}

pub fn compare(_a: &Version, _b: &Version) -> Ordering {
    todo!("compare")
}

pub fn bump_major(_v: &Version) -> Version {
    todo!("bump_major")
}

pub fn bump_minor(_v: &Version) -> Version {
    todo!("bump_minor")
}

pub fn bump_patch(_v: &Version) -> Version {
    todo!("bump_patch")
}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {}
}
