"""Task registry -- the complexity ladder.

    from counting.tasks import build_task, ladder
    task = build_task("nested_key_path", negative_fraction=0.5)

`ladder()` returns the registered tasks ordered by level, which is what
`counting list-tasks` prints. Add a rung by importing the class and adding one
line to TASKS.
"""

from __future__ import annotations

from typing import Any

from .base import Task
from .count_with_load import CountWithLoadTask
from .dual_task import DualTaskCountTask
from .grid_rule_execution import GridRuleExecutionTask
from .formatted_count import FormattedCountTask
from .homogeneous_count import HomogeneousCountTask, MotifCountTask
from .nested_key_path import NestedKeyPathTask
from .rule_execution import RuleExecutionTask
from .run_structure import RunStructureTask
from .state_dual import StateDualTask, StateSoloTask

TASKS: dict[str, type[Task]] = {
    HomogeneousCountTask.name: HomogeneousCountTask,
    MotifCountTask.name: MotifCountTask,
    NestedKeyPathTask.name: NestedKeyPathTask,
    DualTaskCountTask.name: DualTaskCountTask,
    StateDualTask.name: StateDualTask,
    StateSoloTask.name: StateSoloTask,
    RunStructureTask.name: RunStructureTask,
    RuleExecutionTask.name: RuleExecutionTask,
    GridRuleExecutionTask.name: GridRuleExecutionTask,
    CountWithLoadTask.name: CountWithLoadTask,
    FormattedCountTask.name: FormattedCountTask,
}

# Names used by the JSON configs that predate the package layout.
ALIASES = {
    "count": "homogeneous_count",
    "counting": "homogeneous_count",
    "baseline": "homogeneous_count",
    "motif": "motif_count",
    "key_path_match": "nested_key_path",
    "nested": "nested_key_path",
    "dual_task": "dual_task_count",
}


def resolve_task_name(name: str) -> str:
    key = str(name).strip().lower()
    return ALIASES.get(key, key)


def build_task(name: str, **options: Any) -> Task:
    key = resolve_task_name(name)
    if key not in TASKS:
        known = ", ".join(sorted(TASKS))
        raise KeyError(f"unknown task {name!r}; known tasks: {known}")
    return TASKS[key](**options)


def ladder() -> list[type[Task]]:
    """Registered tasks ordered by complexity level, then name."""
    return sorted(TASKS.values(), key=lambda task: (task.level, task.name))


__all__ = ["TASKS", "Task", "build_task", "ladder", "resolve_task_name"]
