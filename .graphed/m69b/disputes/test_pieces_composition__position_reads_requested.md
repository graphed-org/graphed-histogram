# Test Dispute — `tests/frozen/m69b/test_pieces_composition.py` `_position` (both `test_merged_fills_composed_with_a_counter_each_read_once_per_marked_fill` cases)

Status: CLOSED — owner ruling via team-lead 2026-11-20: "#228 option A with refreeze authorized" (graphed#228), refrozen in `test(frozen): m69b composed counter reads its position among the plan's outputs`, tag freeze-m69b-fixup8.

## The test

`_position` finds the counter at its index among the compiled IR's outputs, and `Composed.__call__` reads `values[that index]`.

## The clause it contradicts

graphed#228 option A: `reduce` receives one value per output passed, in order, at every opt level; outputs that compile to one node are evaluated once and repeated. With the counter marked after two fills that merge, the compiled index is 1 but the counter's value is at 2 (`counter-last` fails, reading a fill's array as the count).

## Correction

`_position` returns `compiled.requested.index(node.node_id)`, the counter's first index among the outputs passed. The module docstring says the fills are read at their positions among the plan's outputs. No assertion changes.
