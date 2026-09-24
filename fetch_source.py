#!/usr/bin/env python3
"""Download the Python source the class will translate.

Run this ONCE and commit the result, so nobody needs network access in class.

Source: python-semver/python-semver, BSD-3-Clause.
https://github.com/python-semver/python-semver
"""
import urllib.request, pathlib, sys

RAW = "https://raw.githubusercontent.com/python-semver/python-semver/master"
FILES = {
    "src/semver/version.py": "reference/version.py",
    "src/semver/_types.py":  "reference/_types.py",
    "tests/test_parsing.py": "reference/test_parsing.py",
    "tests/test_compare.py": "reference/test_compare.py",
    "tests/test_bump.py":    "reference/test_bump.py",
}

def main():
    here = pathlib.Path(__file__).parent
    for remote, local in FILES.items():
        dest = here / local
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{RAW}/{remote}"
        print(f"  {url}\n    -> {local}", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                dest.write_bytes(r.read())
        except Exception as e:
            print(f"    FAILED: {e}", file=sys.stderr)
            return 1
    (here / "reference" / "LICENSE-NOTICE.txt").write_text(
        "reference/ contains files from python-semver/python-semver,\n"
        "licensed BSD-3-Clause. See https://github.com/python-semver/python-semver\n")
    print("\nDone. Only reference/version.py must be translated; the test files\n"
          "are there so you can see the behaviour that is expected of it.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
