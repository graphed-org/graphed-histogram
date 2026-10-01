# `tests/frozen/m69b` — traceability

m69b (histserv backend): a `gh.boost.Histogram` backed by a `graphed_histogram.histserv.Context` is filled on a
`histserv` 0.2.1 server instead of riding the reduction tree; the plan carries one `ServiceSpec` per packed server
and the value carries a `Receipt` per backed slot until `resolve_services`/`unpack` snapshots it. Authority: the
services plan (`plan-services.md`, the graphed-histogram frozen table, its "Fails on" list and the sizing,
packing, server-spec and served-process clauses beside it). Frozen from the freeze tag: read-only, never edited,
skipped, xfailed or weakened. A test that looks wrong is a Test Dispute at `.graphed/m69b/disputes/<test_id>.md`.

The tests reach the implementation only through `histserv_harness.histserv_api()` / `boost_api()`
(`importlib.import_module`), so the suite collects before it exists. `histserv_harness.py` starts
`python -m histserv` subprocesses on never-reused free ports and kills them at teardown, copies `run_bounded`
(a hung run fails instead of stalling), and holds the size model (`predicted`, constants `BASE`, `PER_HIST`,
`PER_TASK`, `FILL_A`, `FILL_B`, `PER_CONN` = the plan's `B, O, I, a, b, K`) and Linux `peak(pid)` (`VmHWM` after
writing 5 to `/proc/<pid>/clear_refs`). Every context name is unique to its test except where the test is about
sharing. A refusal is any of `GraphedError`, `TypeError`, `ValueError`. The twin of a backed plan is the same
histograms unbacked.

| Test | Plan clause | Shows | Fails on |
|---|---|---|---|
| **test_histserv_surface.py** (no server) | | | |
| `..._is_a_deferred_histogram_and_backed_returns_its_argument` | `histserv.Histogram` is a `gh.boost.Histogram`; `backed(h, ctx) -> h` | subclass, identity return | a wrapper type |
| `test_backing_leaves_the_graph_identity_of_the_unbacked_twin` | backing is outside graph identity | IR bytes and `external_key`s equal across constructed / backed before fill / after fill / unbacked | the backing in graph identity |
| `test_a_growth_axis_is_refused_naming_histserv_and_the_axis` [Regular, StrCategory, IntCategory] | a growth axis cannot be sized | refusal naming histserv and the axis type | an unsizable axis accepted |
| `test_a_storage_histserv_does_not_hold_is_refused_naming_it` [Mean, WeightedMean, Unlimited, AtomicInt64] | only Double/Int64/Weight | refusal naming histserv and the storage | an unsizable storage accepted |
| `test_category_and_boolean_axes_are_accepted` [Double, Int64, Weight] | every other recorded axis is accepted | StrCategory × IntCategory × Boolean backs | an over-broad refusal |
| `test_the_single_histogram_plan_refuses_a_backed_histogram` | `Histogram.plan()` refuses a backed histogram | refusal | a backed histogram silently planned unserved |
| `test_backing_and_planning_import_neither_histserv_nor_grpc` | no import before a bound run | child's `sys.modules` lacks both; control: `histserv` importable | an eager `histserv` import |
| `test_an_unbacked_merge_free_plan_equals_...folded_in_partition_order` | unbacked `gh.plan` unchanged, `services == ()` | bit-for-bit vs a direct per-partition `bh` fold | an unbacked plan perturbed by the serve path |
| `test_the_frozen_suites_before_m69b_are_unmodified_but_for_the_refreeze` | earlier frozen suites unmodified | per-directory sha256 of m23…m64 (m48/m49 at the refreeze) | an edited frozen suite |
| `test_histserv_imports_wherever_the_gil_is_enabled` | `import histserv` where the GIL is enabled | skipped only on free-threaded builds | a broken histserv dependency |
| **test_histserv_packing.py** (no server) | | | |
| `test_slots_land_first_fit_in_stored_then_key_order_with_the_model_prediction` | sort (`stored` desc, `str(slot)`), (a) first open server it fits; `ctx.servers()`; spec fields and argv | exact slot → server map; each prediction = the model and ≤ size; `ServiceSpec`/`Launch` fields | insertion, ascending or reversed-tie order; a server past its prediction; a missing or mis-scaled model term |
| `test_with_one_size_the_server_count_rises_with_workers_and_tasks_and_falls_with_the_size` | prediction terms in `workers`, `tasks`, size | strictly monotone server counts | a missing model term |
| `test_one_context_with_two_sizes_opens_the_smallest_that_fits_and_never_refuses_on_size` | (b) smallest offered size it fits, (c) own server at the alone prediction rounded up to the MiB holding no other slot | `memory_mb` sorted; exact map; sizes ⌈alone⌉ / `B`+32 / `B`+11 MiB; `resources["memory_mb"]` | a slot refused for its size; an oversize server reused; largest-size-first |
| `test_two_contexts_in_one_plan_never_share_a_server` | each context packs only its own slots | X's slot opens an X server sized to it though Y's server has room | a slot on another context's server |
| `test_a_slot_past_the_message_ceiling_is_refused_naming_the_sizes` | `stored + 64 KiB > 2^29` refused naming slot and sizes | boundary: exactly at the ceiling accepted, one bin over refused | a missing or off-by-one ceiling |
| `test_an_adaptive_plan_a_repeated_partition_and_a_second_serve_are_refused` | `next_tasks`, one partition in two tasks, served twice; only with a backed slot | three refusals; unbacked plans pass through | a retry key that cannot be trusted |
| `test_one_context_over_two_plans_fills_the_open_servers_first_and_collate_starts_each_once` | (a) includes earlier serves' servers; `collate` | B's slot lands on A's last server; collate names each server once | packing per plan |
| `test_an_equal_second_context_shares_the_first_ones_servers_without_overfilling` | equal contexts share servers and packing, with a warning | warning names the name; recomputed predictions ≤ size | equal contexts packing one server twice |
| `test_a_name_holds_its_arguments_for_the_process` / `test_a_later_test_still_finds_the_name_held` | a name holds its arguments for the process; sizes are a set; empty or below `B` refused | refusal naming both sizes; `[s, L]`/`[L, s]` equal; held in a later test (file order) | a per-instance or per-test name registry |
| `test_plan_services_are_sorted_by_name` | `plan.services` sorted by name | `-a-0` before `-z-0` regardless of slot order | insertion-ordered services |
| `test_the_two_size_plan_packs_identically_under_any_hash_seed` | deterministic packing | pickled `(services, slots)` equal under `PYTHONHASHSEED` 1 and 2 | packing by hash order |
| **test_histserv_fill_path.py** | | | |
| `test_receipts_ride_the_tree_and_each_server_holds_exactly_its_slots` | the reduce ships one FillMany per slot and partition; receipts; `unpack` keeps, `resolve_services` deletes | receipts < 1 KiB name the assigned endpoint (twins > 64 KiB); per-server `stats()` count and bytes; 0 Snapshots; `unpack` twice = twin incl. category overflow bins; resolve empties servers; the oversize slot alone on its server | a histogram riding the tree; a snapshot per partition; a slot off its assigned server; a category overflow lost; a deleting `unpack`; a resolve leaving server copies; a slot refused for its size |
| `test_server_names_bound_to_one_endpoint_run_on_one_server` | several names may share one endpoint | distinct `hist_id`s, values = twin | a name-keyed server collision |
| `test_a_tls_or_http_endpoint_is_refused_and_never_dialled` [grpcs, https, http] | plaintext gRPC only | refusal at bind; the listener never accepts | a TLS/HTTP endpoint dialled |
| **test_histserv_retries.py** | | | |
| `test_a_partition_run_twice_fills_once_and_another_partition_adds` | `unique_id=str(partition)`; `ALREADY_EXISTS` is done | `was_filled_with_unique_id`; same `hist_id`; one fill's contents, then the next partition adds | a retry counted twice |
| `test_a_killed_server_raises_a_histserv_error_that_survives_a_pickle` | `HistservError(endpoint, code, details)`, picklable | `UNAVAILABLE` and the address, equal after a round trip | a lost or unpicklable error |
| `test_receipts_of_two_server_histograms_refuse_to_add` | receipts add by identity | same id adds; different ids refused naming both | two server histograms silently merged |
| **test_histserv_lazy_init.py** | | | |
| `test_require_bound_names_every_server_and_accepts_names_sharing_an_endpoint` | `UnboundService` names every server | `names` = all servers; one shared endpoint binds | a partial unbound check |
| `test_histograms_are_created_on_the_first_call_once_and_on_a_bound_pickle_once` | `_Handles` init once, under a lock, on first call or first driver pickle | Init counts 0 after bind, assigned after first call, none for a `replace`d copy or unbound pickle, once for a double pickle | a server histogram created at bind |
| `test_a_process_pool_creates_once_and_equals_the_local_run` | workers receive created ids | Init counts = assigned under `ProcessPoolExecutor(2)`; values = local | a server histogram created in a worker |
| `test_nothing_imports_histserv_before_a_bound_run_does` | no import before a bound run | `before == []`, control `after` True | an eager `histserv` import |
| **test_pieces_composition.py** | | | |
| `test_a_composed_served_plan_equals_gh_plan_and_counts_events` | `pieces` composes with another output | histograms = `gh.plan`'s; counter right | fills read at marked order |
| `test_an_unserved_composed_plan_fails_its_first_task_naming_serve` | a backed reduce outside a served task refuses | raises naming `serve` | an unserved plan run silently |
| `test_a_reduce_whose_hook_never_ran_fails_naming_on_compiled` | positions come from `on_compiled` | raises naming `on_compiled` | positions guessed |
| `test_one_pieces_feeds_one_plan_and_the_first_still_runs_right` | a second firing refused before recording | raises naming `pieces`; the first plan still = direct fill | one `pieces` feeding two plans |
| `test_gh_plan_of_backed_merged_fills_equals_a_direct_fill_done_twice` | each marked fill read at its compiled position | merged pair = fill done twice | a merged fill read once |
| `test_merged_fills_composed_with_a_counter_each_read_once_per_marked_fill` [counter-first, counter-last] | as above, composed | instrument: 2 nodes → 1 output; backed and local = fill twice; counter right | a merged fill read once or a counter read as a fill |
| `test_collated_served_plans_route_each_plan_to_its_own_servers` | `collate` of served plans | receipts only on own servers; values = each run alone | a slot off its assigned server |
| **test_histserv_memory_model.py** (Linux) | | | |
| 64 MiB Double fills + resolve; 32 MiB with 4 `Barrier`-released fillers; 8-label Weight; 2000 one-bin Weight slots; one slot at `workers=256` dialled by 256 local-subchannel-pool channels | server peak ≤ `B` + Σ(`O + I×tasks + (chunks+1)×dense`) + `(a + b×workers)×M + K×workers` | `VmHWM − warm ≤ prediction − B`; warm `< B`; large ones `> (chunks+1)×dense + M`; overhead row's `O` terms ≥ 90 %; connections `> K×N/2` | a server past its prediction; a missing or mis-scaled model term |
