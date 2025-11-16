"""Task definitions for agent experiments."""

from .research_tasks import (
    ResearchTask,
    SIMPLE_TASKS,
    MODERATE_TASKS,
    COMPLEX_TASKS,
    get_all_tasks,
    get_tasks_by_complexity,
    get_task_by_id,
)

__all__ = [
    "ResearchTask",
    "SIMPLE_TASKS",
    "MODERATE_TASKS",
    "COMPLEX_TASKS",
    "get_all_tasks",
    "get_tasks_by_complexity",
    "get_task_by_id",
]
