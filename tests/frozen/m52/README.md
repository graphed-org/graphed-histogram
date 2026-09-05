# `tests/frozen/m52` — traceability

The `graphed-histogram` half of m52 (nuisance POINTS): C5, one resolution rule in the fill lowering.
Authority is `systematics-design/nuisance-points-design.md` §5.1 / §6-C5 and the m52 decomposition's
§3.3. Frozen from the freeze tag: read-only, never edited, skipped, xfailed or weakened. A test that
looks wrong is a Test Dispute at `.graphed/m52/disputes/<test_id>.md`, never a repair in place.

Anchors are cited by DESIGN SECTION. Every fixture is same-directory (`m52_joint_fill_fixtures.py`);
the `m52_` prefix is this tree's helper-basename rule — `tests/frozen` is collected in ONE pytest
process, so a bare helper name would silently bind another milestone's module.

The corpus is the copy VENDORED under `tests/_corpus` (already on `pythonpath`). Nothing here is
`importorskip`-guarded: a skipped headline gate is a silently discharged milestone.

Every m52-new symbol — `graphed.points` and the `points=` keyword — is reached only from a test
BODY, so this tree COLLECTS against the pre-milestone baseline and fails at RUN time.

| anchor | design § | test |
|---|---|---|
| vary-m52-C5 5.1 | §5.1, §6-C5, §4.6 | `test_joint_fill_resolution.py::test_the_joint_labels_axis_value_member_is_the_shifted_one` — the joint label's axis-value member is the observable's `jes_up` member, asserted as a node-id RELATION between the two named candidates and as executed bin contents against the eager oracle |
| vary-m52-C5 5.1-PC | §6-C5 | `test_joint_fill_resolution.py::test_the_eager_universe_oracle_reproduces_the_one_at_a_time_universes` — the oracle validated on the five universes the lowering already gets right, run without `points=` so it stays live at every stage |
| vary-m52-C5 5.2 | §5.1, §6-C5, §4.6 | `test_joint_fill_resolution.py::test_a_btag_only_labels_axis_value_member_stays_nominal` — the one-coordinate point restricts to the origin and keeps NOMINAL kinematics; with 5.1 this is the discriminating pair |
| vary-m52-C5 5.3 | §5.1, §6.2 | `test_joint_axis_mode_parity.py::test_both_fill_modes_resolve_the_joint_label_alike` — sibling and axis mode agree per universe on a WeightedMean fill exercising all three operand roles |
| vary-m52-C5 5.3-PC | §5.1 | `test_joint_axis_mode_parity.py::test_each_mode_moves_the_joint_universe_off_the_btag_only_one` and `::test_the_fill_carries_all_three_operand_roles` — parity between two identically-wrong modes is excluded, and the three roles are witnessed present in the recorded fill node |
| vary-m52-C4 5.4 | §4.10 | `test_points_bare_histogram.py::test_points_refuses_a_bare_axis_mode_histogram`, with `::test_points_answers_on_the_record_time_shape` as the control that `points()` answers at all |
| vary-m52-C5 5.5 | §5.1, §8-h | `test_single_fallback_rule.py::test_no_second_implementation_of_the_fallback_rule_exists` — the null grep beside BOTH its controls in the same test: one written line per alternation branch through the identical matcher, and the behavioral fallback identity |

## Fixture families (`m52_joint_fill_fixtures.py`)

* `joint_context(source, *, register_point=True)` — the single-carrier program §4.11-4 admits: a
  `jes` SHIFT on the event context, then a `btag` WEIGHT family over the SHIFTED jets whose third
  tag carries the point `{btag: hf_up, jes: up}`. The point is registered through the shift-form
  carrier; the loose form carries no `jes` tag and the reachability check refuses a coordinate no
  carrier registers. `register_point=False` drops the keyword and leaves the same six labels on
  default points — the shape the oracle control runs against.
* `ambient_only_fill(context)` — one axis value on `Double()` storage, weighted by the ambient b-tag
  weight alone. 5.1/5.2 read the axis-value member off this fill's recorded input prefix.
* `three_role_fill(context, *, axis_mode)` — the same observable on `WeightedMean()` storage with an
  extra weight FACTOR and `sample=`, both `Varied` over `jes`: every operand role the resolution
  rule is reached from. `axis_mode` toggles m50's frozen per-`fill()` opt-in.
* `eager_universe(jes_scale, sf_coeff)` / `UNIVERSE` — the pure-awkward oracle, one row per
  universe. The JOINT row carries BOTH coordinates: shifted kinematics AND the heavy-flavour SF
  evaluated at the shifted pT.

## Traps this tree is written around

* the ambient weight applies AUTOMATICALLY — passing `weight=[graphed.weight(context)]` back into
  the fill applies it a second time and squares the SF;
* the b-tag SF's coefficient multiplies each JET's factor INSIDE the product, so the SF inherits the
  jet pT of whichever universe evaluates it — that pT dependence is what makes the joint universe
  differ from the b-tag-only one at all;
* `hf_up` is bound ONCE and passed as the member of two tags, so the two labels enter the fill with
  the same recorded SF expression and only the axis-value member can distinguish them;
* node ids are fixture-dependent and no integer is pinned — every structural assertion is a relation
  between two named candidate members;
* axis mode has no in-memory route: its results come from `graphed_histogram.plan` through a
  `SequentialRunner`, and its per-label slice is a POSITIONAL `bh` slice, the named-dict form being
  a `TypeError` on a bare `bh.Histogram`;
* the 5.5 rule-grep resolves its tree from `graphed_histogram.__file__`, never a path literal, and
  its control is a written fixture rather than a spelling in `graphed`'s own source — this milestone
  rewrites the canonical `_member_for`, and a spelling control there would redden a frozen test in
  this repository for a legal implementation.
