# Translation results

The Rust implementation now supplies strict parsing, formatting, precedence,
and all three bump operations. It uses only the standard library. The supplied
`rust/src/main.rs`, `Cargo.toml`, and `evaluate.py` are unchanged.

## 1. Measured results and limitations

The default practice evaluation (`seed=0`, `n=300`) builds successfully and
passes **2,425/2,425 differential cases (100%)**, the full precedence chain,
and **21/21 Rust tests**. Those Rust tests also pass in the debug profile.
All reported quality counters for forbidden constructs, clones, unwraps, and
extra dependencies are zero. The Python agent's **19 offline tests** pass.
The machine-readable result is `logs/evaluation-practice.json`.

Additional unchanged-evaluator runs with 1,000 random cases per family report
99.74% for seed 73 and 99.88% for seed 2026. Their generator sometimes places
invalid numeric prereleases, such as `11.30.5-08`, in the valid pool. Python
semver also rejects those inputs; comparison/bump checks then raise checker
errors. Checking each generated command against Python, including expected
invalid-input errors, gives **15,350/15,350 matches** across those two seeds.
See `logs/evaluation-seed-*.json` and `logs/oracle-crosscheck.json`.

The scaffold's core fields are `u64`, unlike Python's unbounded integers.
Oversized core numbers return an error. The infallible bump API saturates at
`u64::MAX`; behavior at that boundary differs from Python and is documented
and tested. Numeric prerelease comparison supports arbitrarily long strings.

## 2. Agent versus scaffold contribution

The completed Python agent provides OpenAI calls, bounded context, source and
oracle tools, logging, termination checks, and checkpoint rollback. However,
this measured Rust implementation was written directly during the interactive
Codex session. **0% of this translation came from a live `agent.py` execution**;
it should not be attributed to an autonomous run of that loop. The existing
scaffold supplied the protocol, reference source, and evaluator. A genuine
`logs/run-*.jsonl` model trajectory still requires a configured OpenAI key/model
and an actual run; offline mocked tests are not a submission trajectory.

## 3. Unintended behavior

There is no live Python-agent behavior to report yet. Broader validation did
expose the generator issue above. The implementation continues to reject those
invalid versions instead of changing SemVer behavior to accommodate mislabeled
test cases.

## 4. Challenges and resolutions

Parsing distinguishes leading zeros in numeric prereleases from legal build
zeros and alphanumeric identifiers. Precedence compares digit strings by length
and lexical order, avoiding integer overflow and temporary allocations. Bumps
always increment, even for prereleases, and discard both metadata fields.
Reference-derived tests cover these rules, all ordered pairs of the precedence
chain, whitespace/Unicode rejection, round trips, and integer boundaries.

Reproduce the practice result after activating `.venv` and Cargo:

```sh
source "$HOME/.cargo/env"
source .venv/bin/activate
python evaluate.py
```
