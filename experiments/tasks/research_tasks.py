"""Research task definitions for agent experiments.

This module provides a collection of research questions designed to test
different allocation strategies across various complexity levels.
"""

from dataclasses import dataclass


@dataclass
class ResearchTask:
    """A research task for agent evaluation.

    Attributes:
        id: Unique task identifier
        question: The research question to ask the agent
        complexity: Task complexity level (simple, moderate, complex)
        expected_tool_use: Expected number of tool calls for this task
        category: Task category (science, business, technology, etc.)
    """

    id: str
    question: str
    complexity: str
    expected_tool_use: int
    category: str

    def __repr__(self) -> str:
        """String representation of task."""
        return f"ResearchTask(id={self.id}, complexity={self.complexity}, category={self.category})"


# Task catalog organized by complexity
SIMPLE_TASKS = [
    ResearchTask(
        id="simple_01",
        question="What is quantum computing and what are its main applications?",
        complexity="simple",
        expected_tool_use=1,
        category="technology",
    ),
    ResearchTask(
        id="simple_02",
        question="What are the main types of renewable energy sources?",
        complexity="simple",
        expected_tool_use=1,
        category="science",
    ),
    ResearchTask(
        id="simple_03",
        question="What is machine learning and how is it different from traditional programming?",
        complexity="simple",
        expected_tool_use=1,
        category="technology",
    ),
]

MODERATE_TASKS = [
    ResearchTask(
        id="moderate_01",
        question=(
            "Compare the efficiency and environmental impact of solar, wind, "
            "and hydroelectric power generation. Which is most suitable for urban areas?"
        ),
        complexity="moderate",
        expected_tool_use=2,
        category="science",
    ),
    ResearchTask(
        id="moderate_02",
        question=(
            "How are large language models like GPT-4 and Claude being used in healthcare? "
            "What are the main benefits and ethical concerns?"
        ),
        complexity="moderate",
        expected_tool_use=2,
        category="technology",
    ),
    ResearchTask(
        id="moderate_03",
        question=(
            "Explain how blockchain technology works and analyze its potential "
            "applications beyond cryptocurrency."
        ),
        complexity="moderate",
        expected_tool_use=2,
        category="technology",
    ),
]

COMPLEX_TASKS = [
    ResearchTask(
        id="complex_01",
        question=(
            "Analyze the economic and environmental tradeoffs of transitioning "
            "to electric vehicles in developing countries. Consider infrastructure "
            "requirements, battery production impacts, and grid capacity limitations."
        ),
        complexity="complex",
        expected_tool_use=3,
        category="economics",
    ),
    ResearchTask(
        id="complex_02",
        question=(
            "How is quantum computing expected to impact cryptography and cybersecurity "
            "in the next decade? What are the main challenges and potential solutions "
            "for post-quantum encryption?"
        ),
        complexity="complex",
        expected_tool_use=3,
        category="technology",
    ),
    ResearchTask(
        id="complex_03",
        question=(
            "Examine the role of AI in drug discovery and development. Compare recent "
            "successes with traditional methods, discuss regulatory challenges, and "
            "evaluate the potential for personalized medicine."
        ),
        complexity="complex",
        expected_tool_use=3,
        category="healthcare",
    ),
]


def get_all_tasks() -> list[ResearchTask]:
    """Get all research tasks.

    Returns:
        List of all available research tasks
    """
    return SIMPLE_TASKS + MODERATE_TASKS + COMPLEX_TASKS


def get_tasks_by_complexity(complexity: str) -> list[ResearchTask]:
    """Get tasks filtered by complexity level.

    Args:
        complexity: Complexity level (simple, moderate, complex)

    Returns:
        List of tasks matching the complexity level

    Raises:
        ValueError: If complexity level is invalid
    """
    complexity_map = {
        "simple": SIMPLE_TASKS,
        "moderate": MODERATE_TASKS,
        "complex": COMPLEX_TASKS,
    }

    if complexity not in complexity_map:
        raise ValueError(
            f"Invalid complexity: {complexity}. "
            f"Must be one of: {list(complexity_map.keys())}"
        )

    return complexity_map[complexity]


def get_task_by_id(task_id: str) -> ResearchTask | None:
    """Get a specific task by its ID.

    Args:
        task_id: Task identifier

    Returns:
        ResearchTask if found, None otherwise
    """
    for task in get_all_tasks():
        if task.id == task_id:
            return task
    return None
