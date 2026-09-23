# `tests/frozen/m64` — traceability

m64 (histogram growth): `StrCategory([], growth=True)`, `IntCategory([], growth=True)`,
`Integer(..., growth=True)` and `Regular(..., growth=True)` fill over a partitioned source and
combine to D5's reference. Authority: the lane plan
`graphed-workdir/lanes/histogram/plan.md` ("Frozen-test contract", "Reference and deviations
(D5)", "Owner ruling (2026-09-23)") and `.graphed/m64/disputes/`. Frozen from the tag
`freeze-m64`: read-only, never edited, skipped, xfailed or weakened. A test that looks wrong is a
Test Dispute at `.graphed/m64/disputes/<test_id>.md`.

Fixtures live in `m64_growth_fixtures.py` (the `m64_` prefix keeps it from binding another
milestone's helper; spawned workers import it). `ArraySource` sleeps `(n - i) * 0.1` s in
partition `i` on the process runners, so completion order differs from key order; it stamps each
partition's completion time and records whole-dataset loader calls as marker files in a stamp
directory, so both are visible from every worker process. `reference(empty, chunks)` is D5's fold
with `gh.add_histograms`. "All runners" = `SequentialRunner`, `ProcessExecutor(max_workers=2)`,
`ProcessPoolExecutor(max_workers=3)`, each through `h.plan` and `gh.plan`; "two runners" =
`SequentialRunner` and `ProcessPoolExecutor(max_workers=3)` through `gh.plan`. "sw" = the
scramble witness: some partition completed before a lower-keyed one.

Nothing here is `importorskip`-guarded.

| clause | test | at `main` (pre-m64) |
|---|---|---|
| G1 fixed specs keep their bytes | `test_m64_spec_growth.py::test_g1_fixed_specs_keep_their_bytes` | guard, green |
| G2 growth specs (4 axes x 7 storages) | `test_m64_spec_growth.py::test_g2_growth_specs_round_trip` | red |
| G3 unknown versions refused | `test_m64_spec_growth.py::test_g3_unknown_versions_are_refused` | guard, green |
| G4 codec refuses what it cannot carry | `test_m64_spec_growth.py::test_g4_the_codec_refuses_what_it_cannot_carry` (red: DID NOT RAISE); admitted members `::test_g4_the_codec_admits_what_it_carries` | red / guard |
| G5 categories, disjoint and overlapping (brief's proof), loader never called on any runner, sw | `test_m64_category_growth.py::test_g5_growth_categories_equal_one_eager_fill` | red |
| G6 growing `Integer` | `test_m64_category_growth.py::test_g6_growing_integer_equals_one_eager_fill` | red |
| G7 growing `Regular`, dyadic, disjoint and overlapping (brief's proof), sw | `test_m64_regular_growth.py::test_g7_growing_regular_on_a_dyadic_grid_equals_one_eager_fill` | red |
| G8 growing `Regular`, non-dyadic; witness: a partition's grid is off the declared exact grid | `test_m64_regular_growth.py::test_g8_growing_regular_on_a_non_dyadic_grid` | red |
| G9 (i) flows | `test_m64_regular_growth.py::test_g9_flows_stay_in_flow_bins` | red |
| G9 (ii) edge and non-finite values follow `reference` (witness: it differs from one eager fill) | `test_m64_regular_growth.py::test_g9_edge_and_non_finite_values_follow_the_reference` | red |
| G10 2-D, sw | `test_m64_regular_growth.py::test_g10_two_growth_axes` | red |
| G11 `Weight` / `Mean` / `WeightedMean` | `test_m64_storages.py::test_g11_storages_on_growth_axes` | red |
| G12 (a) fixed axes still refuse | `test_m64_add_histograms.py::test_g12a_fixed_axes_still_refuse_to_merge` | guard, green |
| G12 (b) disjoint growing extents, 7 storages, metadata, no mutation | `test_m64_add_histograms.py::test_g12b_disjoint_growing_regular_extents` | red |
| G12 (c) associativity | `test_m64_add_histograms.py::test_g12c_the_merge_is_associative` | red |
| G12 (d) left-first category order | `test_m64_add_histograms.py::test_g12d_category_order_is_left_first` | guard, green |
| G12 (e) misaligned pairs refused | `test_m64_add_histograms.py::test_g12e_misaligned_growing_regulars_are_refused` | guard, green (catches an over-admitting union) |
| G13 two fill calls, partition-major | `test_m64_fill_nodes.py::test_g13_two_fill_calls_are_partition_major` | red |
| G13 axis-mode value variation, partition-major | `test_m64_fill_nodes.py::test_g13_an_axis_mode_value_variation_is_partition_major` | red |
| G14 weight variations, sibling and axis mode | `test_m64_fill_nodes.py::test_g14_sibling_weight_variations_keep_eager_categories`, `::test_g14_axis_mode_weight_variations_keep_eager_categories` | red |
| G15 hist names, parquet, UHI JSON | `test_m64_hist_uhi.py::test_g15_hist_names_growth_and_uhi_round_trip` | red |
| G16 pool rerun determinism under reversed completion; witness: left and right folds differ | `test_m64_determinism.py::test_g16_pool_reruns_are_bitwise_identical` | red |
