"""HotpotQA dataset loader and task definitions for Part 2 multi-agent study.

This module provides utilities for loading, sampling, and working with the
HotpotQA dataset for multi-agent coordination experiments.

HotpotQA: Multi-hop question answering dataset requiring reasoning chains.
- Bridge questions: Require chaining multiple facts
- Comparison questions: Require comparing two entities
"""

import random
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass
class HotpotQATask:
    """A HotpotQA question for multi-agent evaluation.

    Attributes:
        id: Unique task identifier
        question: The multi-hop question to answer
        answer: Ground truth answer
        question_type: Type of question ("bridge" or "comparison")
        level: Difficulty level (all are "hard" in HotpotQA)
    """

    id: str
    question: str
    answer: str
    question_type: str
    level: str

    def __repr__(self) -> str:
        """String representation of task."""
        return f"HotpotQATask(id={self.id}, type={self.question_type})"


class HotpotQALoader:
    """Loader for HotpotQA dataset."""

    def __init__(self, cache_dir: str | None = None):
        """Initialize loader.

        Args:
            cache_dir: Optional directory to cache dataset
        """
        self.cache_dir = cache_dir
        self._dataset: list[HotpotQATask] | None = None

    def load(self) -> list[HotpotQATask]:
        """Load the full HotpotQA validation dataset.

        Returns:
            List of HotpotQATask objects
        """
        if self._dataset is None:
            # Load from Hugging Face (validation split)
            ds = load_dataset("hotpot_qa", "distractor", cache_dir=self.cache_dir)
            self._dataset = self._convert_dataset(ds["validation"])

        return self._dataset

    def _convert_dataset(self, hf_dataset: Any) -> list[HotpotQATask]:
        """Convert Hugging Face dataset to our task format.

        Args:
            hf_dataset: Hugging Face dataset object

        Returns:
            List of HotpotQATask objects
        """
        tasks = []
        for idx, item in enumerate(hf_dataset):
            task = HotpotQATask(
                id=item["id"],
                question=item["question"],
                answer=item["answer"],
                question_type=item["type"],
                level=item["level"],
            )
            tasks.append(task)

        return tasks

    def get_by_type(self, question_type: str) -> list[HotpotQATask]:
        """Get all questions of a specific type.

        Args:
            question_type: Either "bridge" or "comparison"

        Returns:
            List of tasks of that type
        """
        dataset = self.load()
        return [task for task in dataset if task.question_type == question_type]

    def sample_balanced(self, n: int, seed: int | None = None) -> list[HotpotQATask]:
        """Sample questions with balanced representation of both types.

        Args:
            n: Total number of questions to sample
            seed: Random seed for reproducibility

        Returns:
            List of sampled HotpotQATask objects
        """
        dataset = self.load()

        # Set random seed
        if seed is not None:
            random.seed(seed)

        # Separate by type
        bridge_tasks = [t for t in dataset if t.question_type == "bridge"]
        comparison_tasks = [t for t in dataset if t.question_type == "comparison"]

        # Calculate samples per type (aim for 50/50 split)
        n_bridge = n // 2
        n_comparison = n - n_bridge

        # Ensure we don't exceed available samples
        n_bridge = min(n_bridge, len(bridge_tasks))
        n_comparison = min(n_comparison, len(comparison_tasks))

        # Sample from each type
        sampled_bridge = random.sample(bridge_tasks, n_bridge)
        sampled_comparison = random.sample(comparison_tasks, n_comparison)

        # Combine and shuffle
        sampled = sampled_bridge + sampled_comparison
        random.shuffle(sampled)

        return sampled

    def get_type_distribution(self) -> dict[str, int]:
        """Get distribution of question types in the dataset.

        Returns:
            Dictionary mapping question type to count
        """
        dataset = self.load()
        distribution: dict[str, int] = {}
        for task in dataset:
            distribution[task.question_type] = (
                distribution.get(task.question_type, 0) + 1
            )
        return distribution


def get_pilot_sample(seed: int = 42) -> list[HotpotQATask]:
    """Get a pilot sample of 5 questions for initial testing.

    Mix of bridge and comparison question types.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of 5 balanced HotpotQATask objects
    """
    loader = HotpotQALoader()
    return loader.sample_balanced(n=5, seed=seed)


def get_main_sample(n: int = 20, seed: int = 42) -> list[HotpotQATask]:
    """Get main study sample for full experiments.

    Args:
        n: Number of questions to sample (default: 20)
        seed: Random seed for reproducibility

    Returns:
        List of balanced HotpotQATask objects
    """
    loader = HotpotQALoader()
    return loader.sample_balanced(n=n, seed=seed)


if __name__ == "__main__":
    # Test the loader
    print("Loading HotpotQA dataset...")
    loader = HotpotQALoader()
    dataset = loader.load()

    print(f"\nTotal questions: {len(dataset)}")

    # Show type distribution
    distribution = loader.get_type_distribution()
    print("\nQuestion type distribution:")
    for q_type, count in distribution.items():
        print(f"  {q_type}: {count}")

    # Show sample questions
    print("\nSample bridge question:")
    bridge_tasks = loader.get_by_type("bridge")
    if bridge_tasks:
        sample = bridge_tasks[0]
        print(f"  ID: {sample.id}")
        print(f"  Type: {sample.question_type}")
        print(f"  Level: {sample.level}")
        print(f"  Question: {sample.question}")
        print(f"  Answer: {sample.answer}")

    print("\nSample comparison question:")
    comparison_tasks = loader.get_by_type("comparison")
    if comparison_tasks:
        sample = comparison_tasks[0]
        print(f"  ID: {sample.id}")
        print(f"  Type: {sample.question_type}")
        print(f"  Level: {sample.level}")
        print(f"  Question: {sample.question}")
        print(f"  Answer: {sample.answer}")

    # Test balanced sampling
    print("\nTesting balanced sampling...")
    pilot = get_pilot_sample(seed=42)
    print(f"  Pilot sample size: {len(pilot)}")

    # Count types in sample
    type_counts: dict[str, int] = {}
    for task in pilot:
        type_counts[task.question_type] = type_counts.get(task.question_type, 0) + 1

    print("  Type distribution in pilot:")
    for q_type, count in type_counts.items():
        print(f"    {q_type}: {count}")
