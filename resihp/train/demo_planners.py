"""Named planner instances for ``resihp-run --planner module:attr``."""

from resihp.plan.dp_planner import make_dp_planner

dp_8mb = make_dp_planner(num_microbatches=8)
