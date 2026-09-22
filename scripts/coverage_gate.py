#!/usr/bin/env python3
"""Per-file coverage gate: every source file must independently reach 90% (line+branch).

Replaces the old whole-package `fail_under`, which let one well-tested file hide another
that leans on the rest of the repo to look covered. Reads `coverage json`'s per-file
`summary.percent_covered` (line+branch combined, since the run enables `--cov-branch`).
"""

import json
import sys

THRESHOLD = 90.0


def main(path: str) -> int:
    with open(path) as fh:
        files = json.load(fh)["files"]

    under = []
    print(f"{'file':<50} {'pct':>7}")
    for name in sorted(files):
        pct = files[name]["summary"]["percent_covered"]
        print(f"{name:<50} {pct:6.2f}%")
        if pct < THRESHOLD:
            under.append((name, pct))

    print(f"\n{len(files)} files checked against the {THRESHOLD:.0f}% per-file gate.")
    if under:
        print(f"{len(under)} file(s) below {THRESHOLD:.0f}%:")
        for name, pct in under:
            print(f"  {name}: {pct:.2f}%")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "coverage.json"))
