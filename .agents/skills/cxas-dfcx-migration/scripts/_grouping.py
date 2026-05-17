# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Skill-local re-export of the promoted grouping review TUI.

The implementation lives in :mod:`cxas_scrapi.migration.grouping_review` so the
same TUI is reachable from `MigrationCLI`, notebooks, and any other
caller. Skill scripts continue to ``import _grouping`` so existing
call sites don't need to change.

Note: the post-promotion ``interactive_review`` signature returns
``dict | None`` (the accepted groupings), NOT ``(MigrationIR, dict) |
None``. Callers must run their own ``consolidator.consolidate(...)``
after acceptance — this is what ``MigrationService.run_stage1`` does
internally when ``grouping_callback`` is supplied.
"""

# Re-exports for skill consumers — explicit list so the public surface
# stays scannable.
from cxas_scrapi.migration.grouping_review import (  # noqa: F401
    interactive_review,
    render_diff,
    render_ir_tree,
)

# Consolidator helpers — re-exported so skill code that imports them
# from `_grouping` keeps working without changing call sites.
from cxas_scrapi.migration.structural_consolidator import (  # noqa: F401
    GROUP_NAME_RE,
    StructuralConsolidator,
    consolidate,
    detect_root_key,
    load_grouping,
    member_to_group_map,
    persist_grouping,
    root_group_name,
    validate_groupings,
)
