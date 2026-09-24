//! The required SemVer API, translated from reference/version.py.
//!
//! Core numbers use the scaffold's u64 fields. Parsing rejects larger numbers;
//! infallible bumps saturate at u64::MAX instead of wrapping. Python's unbounded
//! integers cannot be fully represented by this interface. Numeric prerelease
//! identifiers have no machine-integer limit.

use std::cmp::Ordering;

/// Metadata excludes its leading separator. Absent metadata is None.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub prerelease: Option<String>,
    pub build: Option<String>,
}

fn numeric(identifier: &str) -> bool {
    !identifier.is_empty() && identifier.bytes().all(|b| b.is_ascii_digit())
}

fn component(part: Option<&str>) -> Result<u64, String> {
    let text = part.ok_or_else(|| String::from("expected major.minor.patch"))?;
    if !numeric(text) || (text.len() > 1 && text.starts_with('0')) {
        return Err(String::from("invalid core number"));
    }
    text.parse().map_err(|_| String::from("core number exceeds u64"))
}

fn validate_metadata(text: &str, prerelease: bool) -> Result<(), String> {
    for identifier in text.split('.') {
        if identifier.is_empty()
            || !identifier.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
        {
            return Err(String::from("invalid metadata identifier"));
        }
        if prerelease && numeric(identifier) && identifier.len() > 1 && identifier.starts_with('0') {
            return Err(String::from("numeric prerelease has a leading zero"));
        }
    }
    Ok(())
}

/// Parse the strict three-component grammar used by Version.parse().
pub fn parse(s: &str) -> Result<Version, String> {
    // Split build first: hyphens inside build metadata are ordinary characters.
    let (version, build) = match s.split_once('+') {
        Some((version, build)) => {
            validate_metadata(build, false)?;
            (version, Some(build))
        }
        None => (s, None),
    };
    let (core, prerelease) = match version.split_once('-') {
        Some((core, prerelease)) => {
            validate_metadata(prerelease, true)?;
            (core, Some(prerelease))
        }
        None => (version, None),
    };
    let mut parts = core.split('.');
    let major = component(parts.next())?;
    let minor = component(parts.next())?;
    let patch = component(parts.next())?;
    if parts.next().is_some() {
        return Err(String::from("expected exactly three core numbers"));
    }
    Ok(Version {
        major,
        minor,
        patch,
        prerelease: prerelease.map(String::from),
        build: build.map(String::from),
    })
}

/// Render a version, preserving metadata exactly.
pub fn to_string(v: &Version) -> String {
    let mut output = format!("{}.{}.{}", v.major, v.minor, v.patch);
    for (separator, metadata) in [('-', v.prerelease.as_deref()), ('+', v.build.as_deref())] {
        if let Some(text) = metadata.filter(|text| !text.is_empty()) {
            output.push(separator);
            output.push_str(text);
        }
    }
    output
}

fn compare_identifier(a: &str, b: &str) -> Ordering {
    match (numeric(a), numeric(b)) {
        (true, true) => {
            // Length and then lexical order compare arbitrary-size decimal integers.
            // Normalization also handles manually constructed Version values.
            let a = a.trim_start_matches('0');
            let b = b.trim_start_matches('0');
            a.len().cmp(&b.len()).then_with(|| a.cmp(b))
        }
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => a.cmp(b),
    }
}

/// SemVer precedence; build metadata does not participate.
pub fn compare(a: &Version, b: &Version) -> Ordering {
    let core = (a.major, a.minor, a.patch).cmp(&(b.major, b.minor, b.patch));
    if core != Ordering::Equal {
        return core;
    }
    match (
        a.prerelease.as_deref().filter(|s| !s.is_empty()),
        b.prerelease.as_deref().filter(|s| !s.is_empty()),
    ) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(a), Some(b)) => {
            let mut a = a.split('.');
            let mut b = b.split('.');
            loop {
                match (a.next(), b.next()) {
                    (None, None) => return Ordering::Equal,
                    (None, Some(_)) => return Ordering::Less,
                    (Some(_), None) => return Ordering::Greater,
                    (Some(a), Some(b)) => {
                        let ordering = compare_identifier(a, b);
                        if ordering != Ordering::Equal {
                            return ordering;
                        }
                    }
                }
            }
        }
    }
}

/// Increment major, reset minor/patch, and discard metadata. Saturates at u64::MAX.
pub fn bump_major(v: &Version) -> Version {
    Version {
        major: v.major.saturating_add(1),
        minor: 0,
        patch: 0,
        prerelease: None,
        build: None,
    }
}

/// Increment minor, reset patch, and discard metadata. Saturates at u64::MAX.
pub fn bump_minor(v: &Version) -> Version {
    Version {
        major: v.major,
        minor: v.minor.saturating_add(1),
        patch: 0,
        prerelease: None,
        build: None,
    }
}

/// Increment patch even on a prerelease, discarding metadata. Saturates at u64::MAX.
pub fn bump_patch(v: &Version) -> Version {
    Version {
        major: v.major,
        minor: v.minor,
        patch: v.patch.saturating_add(1),
        prerelease: None,
        build: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Cases ported from reference/test_parsing.py.
    #[test]
    fn parse_reference_metadata() -> Result<(), String> {
        for (input, core, prerelease, build) in [
            ("1.2.3-alpha.1.2+build.11.e0f985a", (1, 2, 3), Some("alpha.1.2"), Some("build.11.e0f985a")),
            ("1.2.3-alpha-1+build.11.e0f985a", (1, 2, 3), Some("alpha-1"), Some("build.11.e0f985a")),
            ("0.1.0-0f", (0, 1, 0), Some("0f"), None),
            ("0.0.0-0foo.1+build.1", (0, 0, 0), Some("0foo.1"), Some("build.1")),
            ("1.2.3-rc.0.0+build.0", (1, 2, 3), Some("rc.0.0"), Some("build.0")),
        ] {
            let v = parse(input)?;
            assert_eq!((v.major, v.minor, v.patch), core);
            assert_eq!(v.prerelease.as_deref(), prerelease);
            assert_eq!(v.build.as_deref(), build);
        }
        Ok(())
    }

    #[test]
    fn parse_plain_version() -> Result<(), String> {
        let v = parse("10.20.30")?;
        assert_eq!((v.major, v.minor, v.patch), (10, 20, 30));
        assert_eq!(v.prerelease, None);
        assert_eq!(v.build, None);
        Ok(())
    }

    #[test]
    fn reject_leading_core_and_prerelease_zeros() {
        for text in ["01.2.3", "1.02.3", "1.2.03", "1.2.3-01", "1.2.3-a.00", "1.2.3-0.01"] {
            assert!(parse(text).is_err(), "accepted {text}");
        }
    }

    #[test]
    fn allow_build_zeros_and_alphanumeric_prerelease_zeros() -> Result<(), String> {
        for text in ["1.2.3+001.00", "1.2.3-0", "1.2.3-00a", "1.2.3-01-", "1.2.3-0A.is.legal"] {
            assert_eq!(to_string(&parse(text)?), text);
        }
        Ok(())
    }

    #[test]
    fn reject_malformed_core() {
        for text in ["", "1", "1.0", "1.2.3.4", ".1.2", "1..2", "1.2.", "-1.0.0",
                     "+1.0.0", "1.-1.0", "v1.0.0", "a.b.c", "1e2.0.0"] {
            assert!(parse(text).is_err(), "accepted {text:?}");
        }
    }

    #[test]
    fn reject_malformed_metadata() {
        for text in ["1.2.3-", "1.2.3+", "1.2.3-+a", "1.2.3-a..b", "1.2.3+.a",
                     "1.2.3+a.", "1.2.3+a..b", "1.2.3+a+b", "1.2.3-alpha_beta",
                     "1.2.3-alpha@1", "1.2.3+a/b", "1.2.3-a.+b"] {
            assert!(parse(text).is_err(), "accepted {text:?}");
        }
    }

    #[test]
    fn reject_whitespace_and_non_ascii() {
        // These must be tested directly; the command protocol trims whitespace.
        for text in [" 1.2.3", "1.2.3 ", "1.2.3\n", "1.2.3\r\n", "1.2.\t3",
                     "1.2.3-a b", "1.2.3-é", "1.2.3+β", "١.2.3", "1.2.3-١", "1.2.3\0"] {
            assert!(parse(text).is_err(), "accepted {text:?}");
        }
    }

    #[test]
    fn metadata_can_contain_hyphens() -> Result<(), String> {
        for text in ["1.0.0--hyphen", "1.0.0--+-", "1.0.0+21AF26D3--117B344092BD",
                     "1.0.0+a-b", "1.0.0-a-b+c-d", "1.0.0---.---+--.--"] {
            assert_eq!(to_string(&parse(text)?), text);
        }
        Ok(())
    }

    #[test]
    fn core_integer_boundaries() -> Result<(), String> {
        let maximum = format!("{}.{}.{}", u64::MAX, u64::MAX, u64::MAX);
        assert_eq!(to_string(&parse(&maximum)?), maximum);
        assert_eq!(parse("4294967295.0.0")?.major, 4_294_967_295);
        for text in ["18446744073709551616.0.0", "0.18446744073709551616.0", "0.0.18446744073709551616"] {
            assert!(parse(text).is_err());
        }
        Ok(())
    }

    #[test]
    fn round_trip_preserves_all_fields() -> Result<(), String> {
        for text in ["0.0.0", "1.2.3", "10.20.30-rc.1+001.A-b", "1.0.0+build",
                     "1.0.0-123456789012345678901234567890"] {
            let original = parse(text)?;
            let formatted = to_string(&original);
            assert_eq!(formatted, text);
            assert_eq!(parse(&formatted)?, original);
        }
        Ok(())
    }

    // Cases ported from reference/test_compare.py, including all ordered pairs.
    #[test]
    fn specification_precedence_chain() -> Result<(), String> {
        let chain = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta",
                     "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"];
        for (i, a) in chain.iter().enumerate() {
            for (j, b) in chain.iter().enumerate() {
                assert_eq!(compare(&parse(a)?, &parse(b)?), i.cmp(&j), "{a} vs {b}");
            }
        }
        Ok(())
    }

    #[test]
    fn core_precedence_overrides_metadata() -> Result<(), String> {
        for (a, b) in [("1.99.99", "2.0.0-alpha"), ("1.2.99", "1.3.0-0"),
                       ("1.2.3", "1.2.4-alpha"), ("2.0.0", "10.0.0")] {
            assert_eq!(compare(&parse(a)?, &parse(b)?), Ordering::Less);
            assert_eq!(compare(&parse(b)?, &parse(a)?), Ordering::Greater);
        }
        Ok(())
    }

    #[test]
    fn build_metadata_is_ignored() -> Result<(), String> {
        for (a, b) in [("1.0.0+build.1", "1.0.0"), ("1.0.0-alpha.1+build.1", "1.0.0-alpha.1"),
                       ("1.1.9-rc.1+a", "1.1.9-rc.1+z"), ("1.0.0+001", "1.0.0+999")] {
            assert_eq!(compare(&parse(a)?, &parse(b)?), Ordering::Equal);
            assert_eq!(compare(&parse(b)?, &parse(a)?), Ordering::Equal);
        }
        assert_eq!(compare(&parse("1.1.9-rc.1")?, &parse("1.1.9+build.1")?), Ordering::Less);
        Ok(())
    }

    #[test]
    fn numeric_prerelease_ordering_has_no_integer_limit() -> Result<(), String> {
        for (a, b) in [("2", "11"), ("9", "10"), ("999999999999999999999999999999",
                        "1000000000000000000000000000000"),
                       ("1000000000000000000000000000000", "1000000000000000000000000000001"),
                       ("999999999999999999999999999999", "-"), ("0", "A")] {
            assert_eq!(compare(&parse(&format!("1.0.0-{a}"))?, &parse(&format!("1.0.0-{b}"))?), Ordering::Less);
        }
        Ok(())
    }

    #[test]
    fn prerelease_prefixes_and_ascii_order() -> Result<(), String> {
        for (a, b) in [("alpha", "alpha.0"), ("alpha.1", "alpha.1.0"), ("A", "a"),
                       ("rc0", "rc1"), ("rc10", "rc2"), ("0a", "a")] {
            let a = parse(&format!("1.0.0-{a}"))?;
            let b = parse(&format!("1.0.0-{b}"))?;
            assert_eq!(compare(&a, &b), Ordering::Less);
            assert_eq!(compare(&b, &a), Ordering::Greater);
        }
        Ok(())
    }

    // Cases ported from reference/test_bump.py.
    #[test]
    fn bump_major_resets_lower_components() -> Result<(), String> {
        assert_eq!(to_string(&bump_major(&parse("3.4.5")?)), "4.0.0");
        assert_eq!(to_string(&bump_major(&parse("1.0.0-rc.1")?)), "2.0.0");
        assert_eq!(to_string(&bump_major(&parse("4294967295.0.0")?)), "4294967296.0.0");
        Ok(())
    }

    #[test]
    fn bump_minor_resets_patch() -> Result<(), String> {
        assert_eq!(to_string(&bump_minor(&parse("3.4.5")?)), "3.5.0");
        assert_eq!(to_string(&bump_minor(&parse("0.2.0-rc.1")?)), "0.3.0");
        Ok(())
    }

    #[test]
    fn bump_patch_increments_prereleases_too() -> Result<(), String> {
        for input in ["3.4.5", "3.4.5-rc.1", "3.4.5-rc1+build4", "3.4.5+001"] {
            assert_eq!(to_string(&bump_patch(&parse(input)?)), "3.4.6");
        }
        Ok(())
    }

    #[test]
    fn bumps_discard_metadata_without_mutating_input() -> Result<(), String> {
        let text = "3.4.5-rc.1+build.4";
        let input = parse(text)?;
        for bumped in [bump_major(&input), bump_minor(&input), bump_patch(&input)] {
            assert_eq!(bumped.prerelease, None);
            assert_eq!(bumped.build, None);
        }
        assert_eq!(to_string(&input), text);
        assert_eq!(to_string(&bump_minor(&bump_major(&input))), "4.1.0");
        assert_eq!(to_string(&bump_patch(&bump_minor(&input))), "3.5.1");
        Ok(())
    }

    #[test]
    fn bumps_at_u64_limit_do_not_wrap() -> Result<(), String> {
        let v = parse(&format!("{}.{}.{}-rc+build", u64::MAX, u64::MAX, u64::MAX))?;
        assert_eq!(bump_major(&v).major, u64::MAX);
        assert_eq!(bump_minor(&v).minor, u64::MAX);
        assert_eq!(bump_patch(&v).patch, u64::MAX);
        Ok(())
    }

    #[test]
    fn empty_metadata_on_constructed_values_matches_python() -> Result<(), String> {
        let v = Version {
            major: 1, minor: 2, patch: 3,
            prerelease: Some(String::new()), build: Some(String::new()),
        };
        assert_eq!(to_string(&v), "1.2.3");
        assert_eq!(compare(&v, &parse("1.2.3")?), Ordering::Equal);
        Ok(())
    }
}
