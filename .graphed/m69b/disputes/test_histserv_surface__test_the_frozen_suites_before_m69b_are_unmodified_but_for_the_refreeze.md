# Test Dispute — `tests/frozen/m69b/test_histserv_surface.py::test_the_frozen_suites_before_m69b_are_unmodified_but_for_the_refreeze`

Status: **OPEN** — awaiting an owner ruling.

## The test

`_digest` hashes each pre-m69b frozen directory over `sorted(Path, ...)` and compares it with `FROZEN_BEFORE_M69B`. On every Windows leg of graphed-histogram#24's CI run 36809889213, the digests of m23, m48, m49, m52 and m64 differ. m29 and m50 match.

## The clause it contradicts

§A.5 (the full OS matrix) and plan-services.md §5.1, which pins that the pre-m69b suites are unmodified, not that they hash differently per OS. `WindowsPath` ordering is case-insensitive, so a directory with an upper-case name (`README.md`, in exactly the five failing directories) hashes its files in a different order. Re-hashing locally in `PureWindowsPath` order reproduces CI's five Windows digests exactly (3b599cbd, 282d0acc, 96e0d0cd, 2d49a3df, 48587884). The files are byte-identical; the test fails on an unmodified tree.

## Correction

`sorted(..., key=lambda q: q.relative_to(root).parts)` in `_digest`. On POSIX it reproduces every digest in `FROZEN_BEFORE_M69B` unchanged (checked locally for all seven), and on Windows it orders the files the same way.
