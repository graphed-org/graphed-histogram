# m48 — `vary` weight path, the fill-shaped half

The frozen acceptance suite for milestone m48 in `graphed-histogram`. Every anchor here needs a
`Histogram.fill`, which is partition rule (1) of §10/m48: the rest of m48's anchors live in
`graphed`'s `tests/frozen/frontend/m48` and `tests/frozen/awkward/m48`.

Authority: `systematics-vary-plan.md` r33. Section references in the files are to that plan.

## Traceability

| anchor | plan clause | file :: function |
|---|---|---|
| H1 | §10/m48 corpus weight-variation references; §4.1 flat-SF spelling; §4.2 sibling weight fills | `test_corpus_weight_matrix.py::test_the_weight_matrix_reproduces_its_corpus_reference` |
| H1 | §6.1a absent labels are absent, never duplicated from nominal | `test_corpus_weight_matrix.py::test_every_output_carries_exactly_its_own_labels` |
| H1 | §5.2b single-read witness, bound to the reference-matrix run | `test_corpus_weight_matrix.py::test_the_reference_run_reads_each_partition_exactly_once` |
| H2 | §9.1 per-label fill-node accessor | `test_selection_invariance.py::test_every_label_has_its_own_fill_node_under_the_per_label_accessor` |
| H2 | §4.3 structural selection-invariance (the binding per-label input-prefix form) | `test_selection_invariance.py::test_the_non_weight_input_prefix_is_identical_across_every_weight_label` |
| H2 | §4.3 discriminator: the weight input is what differs | `test_selection_invariance.py::test_the_weight_input_is_what_differs_between_the_labels` |
| H2 | §4.3 m05 equal-counts sanity | `test_selection_invariance.py::test_the_labels_occupy_the_same_bins_and_carry_different_contents` |
| H12 | §6.3(1) committed pre-m48 golden GIR blob, stripped per side | `test_variation_goldens.py::test_the_unvaried_fill_graph_still_serializes_to_the_pre_m48_golden` |
| H12 | §6.3(1) the golden is committed already stripped | `test_variation_goldens.py::test_the_golden_carries_no_version_bytes_of_its_own` |
| H12 | §6.3 params KEY SET against a literally spelled set | `test_variation_goldens.py::test_the_unvaried_single_weight_fill_records_exactly_these_params` |
| H12 | §6.3 closing monkeypatch leg (per-side stripping) | `test_variation_goldens.py::test_the_comparison_survives_a_boost_histogram_version_bump` |

## Spellings pinned at this freeze (§9.1, §4.4 of the decomposition)

| surface | shape |
|---|---|
| `graphed_histogram.unpack(value)` | `dict[str, bh.Histogram \| dict[str, bh.Histogram]]` over the executed plan value alone |
| `graphed_histogram.fill_nodes_by_label(h)` | `dict[str, Array]`, label order per §2.4 (nominal first) |
| `graphed_histogram.plan(...)` value | flat `{output: hist}` for an output no variation reaches, `{(output, label): hist}` for a varied sibling output |
| `Histogram.fill(..., unweighted=True)` | suppresses the ambient weight AND every explicit `weight=[…]` factor |

## Fixtures

`tests/_corpus/` is the vendored `graphed-corpus` (§10 preamble: vendoring, not a dependency, and
not `importorskip`). H1 owns its own read-counting `PartitionedSource` over the references' exact
dataset; the toy fixtures the lowering anchors share live in `vary_hist_fixtures.py`.
