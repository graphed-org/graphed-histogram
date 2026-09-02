"""m49/F2 — §8.2(i): the `variation_labels` POPULATION, at its only bound producer.

`_PartitionReduce.variation_labels` is declared in `graphed` and returned through §7.2's (β)
channel, but the group-plan builder in THIS repo is the only thing that ever fills it — a
hook-less `aggregate_plan` returns `None` whatever a producer does, so no `graphed` tree can
witness either the payload or the `None` rule's admitted member.

The fixture's shared node sits UPSTREAM of the label fork (§3.4's shape): interning keys on input
ids, so no node downstream of the fork can be reached by two labels with distinct members, and by
§6.1b's count distinct labels have distinct FILL nodes — no fill-node key is ever multi-label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graphed.core import GraphStore
from m49_hist_fixtures import shared_node_program

import graphed_histogram as gh

FIXTURES = Path(__file__).resolve().parent / "m49_hist_fixtures.py"
STEPS = 3


def _plans() -> tuple[Any, Any]:
    """One session; a varied builder call and an UNVARIED one on a different chain."""
    _session, varied, plain, _data = shared_node_program()
    return gh.plan({"q": varied}, steps_per_file=STEPS), gh.plan({"plain": plain}, steps_per_file=STEPS)


def _entries() -> tuple[Any, ...]:
    payload = _plans()[0].process.variation_labels
    assert payload is not None, "the varied plan's shipped closure carries no variation labels"
    return tuple(payload)


def _sort_key(key: tuple[int, int | None]) -> tuple[int, int]:
    """§8.2(i)'s bound order. A bare `sorted()` over the keys is a `TypeError` the moment one
    reduced id carries both an indexed and a `None` entry."""
    reduced_id, member_index = key
    return (reduced_id, -1 if member_index is None else member_index)


def _no_unordered(value: object) -> bool:
    if isinstance(value, set | frozenset):
        return False
    if isinstance(value, tuple | list):
        return all(_no_unordered(item) for item in value)
    return True


def test_every_entry_has_the_bound_layout() -> None:
    """`((reduced_node_id, member_index | None), (labels, frame))`, `labels` a sorted tuple of
    strings — a producer in one repo and a consumer in another share this shape."""
    for entry in _entries():
        key, (labels, frame) = entry
        reduced_id, member_index = key
        assert isinstance(reduced_id, int)
        assert member_index is None or isinstance(member_index, int)
        assert isinstance(labels, tuple)
        assert all(isinstance(label, str) for label in labels)
        assert list(labels) == sorted(labels)
        assert frame is not None


def test_the_entries_are_sorted_on_the_bound_key_and_unique() -> None:
    keys = [entry[0] for entry in _entries()]
    assert keys == sorted(keys, key=_sort_key)
    assert len(keys) == len(set(keys))


def test_nothing_in_the_payload_is_a_set_or_frozenset() -> None:
    """A `frozenset` pickles in hash order, so the closure's cloudpickle bytes — and through
    `OpSpec.identity()` every `DurablePlan` fingerprint built over them — would vary with
    `PYTHONHASHSEED`."""
    assert _no_unordered(_entries())


def test_every_key_names_a_node_of_the_reduced_store_shipped_beside_it() -> None:
    """The keying is POST-REDUCTION ids from the SAME compile that produced the shipped `ir`;
    record-time ids are wrong because the reduction re-indexes. The artifact travelling in the
    same closure is the oracle."""
    plan, _plain = _plans()
    node_ids = {node["id"] for node in GraphStore.deserialize(plan.process.ir).nodes()}
    assert node_ids
    payload = plan.process.variation_labels
    assert payload is not None
    assert {key[0] for key, _value in payload} <= node_ids


def test_a_key_two_label_cones_both_reach_carries_BOTH_labels() -> None:
    """The set-valued half of the key space. An implementation that picks one label arbitrarily,
    or that keys per label, cannot produce this entry."""
    both = [labels for _key, (labels, _frame) in _entries() if len(labels) > 1]
    assert ("s_down", "s_up") in both


def test_the_shared_prefix_carries_the_labels_and_never_the_string_nominal() -> None:
    """§8.2(i)'s nominal-exclusion clause: every label's cone reaches the shared prefix, so a
    producer that unions raw label sets would put `nominal` on it."""
    seen: set[str] = set()
    for _key, (labels, _frame) in _entries():
        seen.update(labels)
    assert seen == {"s_down", "s_up"}


def test_a_key_only_the_nominal_cone_reaches_carries_an_EMPTY_tuple_and_a_real_frame() -> None:
    """The empty tuple is how a key no non-nominal label reaches is carried — it renders `""` and
    must still point at the user's line. Dropping such keys leaves the nominal universe's failures
    unattributed."""
    nominal_only = [frame for _key, (labels, frame) in _entries() if labels == ()]
    assert nominal_only
    assert any(FIXTURES.name in str(frame) for frame in nominal_only)


def test_every_frame_points_into_the_module_that_recorded_the_program() -> None:
    frames = [str(frame) for _key, (_labels, frame) in _entries()]
    assert frames
    assert all(FIXTURES.name in frame for frame in frames)


def test_the_payload_is_identical_across_two_independent_builds() -> None:
    """§3.2 reaches the closure through this field: two builds of one program ship one payload."""
    assert _plans()[0].process.variation_labels == _plans()[0].process.variation_labels


def test_an_unvaried_builder_call_in_a_varied_session_ships_None() -> None:
    """The admitted member of the hook's `None` rule (§8.2(i)): the predicate is over the LABELS
    of the compiled program, not over the session. A session-scoped producer populates this one
    too and is red — and only this repo can witness it, since a hook-less `aggregate_plan`
    returns `None` whatever the producer does.

    The paired assertion is the whole file above: the SAME session's varied call must be
    populated, so a producer that simply never fires satisfies neither."""
    varied, plain = _plans()
    assert varied.process.variation_labels is not None
    assert plain.process.variation_labels is None
