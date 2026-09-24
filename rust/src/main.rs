//! Evaluation harness. DO NOT EDIT — evaluate.py depends on this protocol.
//!
//! Reads one command per line on stdin, writes one JSON object per line on
//! stdout. Every command is wrapped in catch_unwind, so a panic in your
//! library is reported as a failed case rather than killing the run.
//!
//!   parse <version>            -> {"ok":true,"major":1,"minor":2,"patch":3,
//!                                  "prerelease":"alpha.1","build":null}
//!   compare <a> <b>            -> {"ok":true,"cmp":-1}          (-1 | 0 | 1)
//!   bump <major|minor|patch> <version>
//!                              -> {"ok":true,"version":"2.0.0"}
//!   format <version>           -> {"ok":true,"version":"1.2.3-alpha.1"}
//!
//! Any failure (invalid input, unsupported op, panic) -> {"ok":false,...}

use py2rust_semver as sv;
use std::io::{self, BufRead, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn jstr(s: Option<&str>) -> String {
    match s {
        None => "null".to_string(),
        Some(v) => format!("\"{}\"", v.replace('\\', "\\\\").replace('"', "\\\"")),
    }
}

fn err(kind: &str) -> String {
    format!("{{\"ok\":false,\"error\":{}}}", jstr(Some(kind)))
}

fn dispatch(parts: &[&str]) -> String {
    match parts {
        ["parse", v] => match sv::parse(v) {
            Ok(x) => format!(
                "{{\"ok\":true,\"major\":{},\"minor\":{},\"patch\":{},\"prerelease\":{},\"build\":{}}}",
                x.major, x.minor, x.patch,
                jstr(x.prerelease.as_deref()), jstr(x.build.as_deref())),
            Err(e) => err(&format!("parse: {e}")),
        },
        ["compare", a, b] => match (sv::parse(a), sv::parse(b)) {
            (Ok(x), Ok(y)) => {
                let c = match sv::compare(&x, &y) {
                    std::cmp::Ordering::Less => -1,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                };
                format!("{{\"ok\":true,\"cmp\":{c}}}")
            }
            _ => err("compare: unparseable operand"),
        },
        ["bump", kind, v] => match sv::parse(v) {
            Ok(x) => {
                let out = match *kind {
                    "major" => Some(sv::bump_major(&x)),
                    "minor" => Some(sv::bump_minor(&x)),
                    "patch" => Some(sv::bump_patch(&x)),
                    _ => None,
                };
                match out {
                    Some(y) => format!("{{\"ok\":true,\"version\":{}}}", jstr(Some(&sv::to_string(&y)))),
                    None => err("bump: unknown kind"),
                }
            }
            Err(e) => err(&format!("bump: {e}")),
        },
        ["format", v] => match sv::parse(v) {
            Ok(x) => format!("{{\"ok\":true,\"version\":{}}}", jstr(Some(&sv::to_string(&x)))),
            Err(e) => err(&format!("format: {e}")),
        },
        _ => err("unknown command"),
    }
}

fn main() {
    let stdin = io::stdin();
    let mut out = io::stdout();
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        let line = line.trim();
        if line.is_empty() { continue; }
        let parts: Vec<&str> = line.split(' ').filter(|s| !s.is_empty()).collect();
        let resp = catch_unwind(AssertUnwindSafe(|| dispatch(&parts)))
            .unwrap_or_else(|_| err("panic"));
        writeln!(out, "{resp}").ok();
        out.flush().ok();
    }
}
