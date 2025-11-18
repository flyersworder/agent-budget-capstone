"""TruthfulQA dataset loader and task definitions for Part 2 experiments.

This module provides utilities for loading, sampling, and working with the
TruthfulQA dataset for budget awareness experiments.
"""

import random
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass
class TruthfulQATask:
    """A TruthfulQA question for agent evaluation.

    Attributes:
        id: Unique task identifier
        question: The question to ask the agent
        best_answer: The correct/truthful answer
        correct_answers: List of acceptable correct answers
        incorrect_answers: List of common incorrect answers
        category: Question category (e.g., "Health", "Law", "Finance")
    """

    id: str
    question: str
    best_answer: str
    correct_answers: list[str]
    incorrect_answers: list[str]
    category: str

    def __repr__(self) -> str:
        """String representation of task."""
        return f"TruthfulQATask(id={self.id}, category={self.category})"


class TruthfulQALoader:
    """Loader for TruthfulQA dataset."""

    def __init__(self, cache_dir: str | None = None):
        """Initialize loader.

        Args:
            cache_dir: Optional directory to cache dataset
        """
        self.cache_dir = cache_dir
        self._dataset: list[TruthfulQATask] | None = None

    def load(self) -> list[TruthfulQATask]:
        """Load the full TruthfulQA dataset.

        Returns:
            List of TruthfulQATask objects
        """
        if self._dataset is None:
            # Load from Hugging Face
            ds = load_dataset("truthful_qa", "generation", cache_dir=self.cache_dir)
            self._dataset = self._convert_dataset(ds["validation"])

        return self._dataset

    def _convert_dataset(self, hf_dataset: Any) -> list[TruthfulQATask]:
        """Convert Hugging Face dataset to our task format.

        Args:
            hf_dataset: Hugging Face dataset object

        Returns:
            List of TruthfulQATask objects
        """
        tasks = []
        for idx, item in enumerate(hf_dataset):
            task = TruthfulQATask(
                id=f"truthful_qa_{idx:04d}",
                question=item["question"],
                best_answer=item["best_answer"],
                correct_answers=item["correct_answers"],
                incorrect_answers=item["incorrect_answers"],
                category=item["category"],
            )
            tasks.append(task)

        return tasks

    def sample_stratified(
        self, n: int, seed: int | None = None
    ) -> list[TruthfulQATask]:
        """Sample questions with stratification by category.

        Args:
            n: Number of questions to sample
            seed: Random seed for reproducibility

        Returns:
            List of sampled TruthfulQATask objects
        """
        dataset = self.load()

        # Set random seed
        if seed is not None:
            random.seed(seed)

        # Group by category
        by_category: dict[str, list[TruthfulQATask]] = {}
        for task in dataset:
            if task.category not in by_category:
                by_category[task.category] = []
            by_category[task.category].append(task)

        # Calculate samples per category (proportional sampling)
        total = len(dataset)
        samples_per_category = {
            cat: max(1, int(n * len(tasks) / total))
            for cat, tasks in by_category.items()
        }

        # Adjust to exactly n samples
        while sum(samples_per_category.values()) > n:
            # Remove from largest category
            max_cat = max(samples_per_category, key=lambda x: samples_per_category[x])
            samples_per_category[max_cat] -= 1

        while sum(samples_per_category.values()) < n:
            # Add to largest category that has room
            max_cat = max(
                [
                    cat
                    for cat in samples_per_category
                    if samples_per_category[cat] < len(by_category[cat])
                ],
                key=lambda c: len(by_category[c]) - samples_per_category[c],
            )
            samples_per_category[max_cat] += 1

        # Sample from each category
        sampled = []
        for cat, n_samples in samples_per_category.items():
            sampled.extend(random.sample(by_category[cat], n_samples))

        # Shuffle final sample
        random.shuffle(sampled)

        return sampled

    def get_by_category(self, category: str) -> list[TruthfulQATask]:
        """Get all questions from a specific category.

        Args:
            category: Category name (e.g., "Health", "Law")

        Returns:
            List of tasks in that category
        """
        dataset = self.load()
        return [task for task in dataset if task.category == category]

    def get_categories(self) -> list[str]:
        """Get list of all categories in the dataset.

        Returns:
            Sorted list of unique category names
        """
        dataset = self.load()
        return sorted(set(task.category for task in dataset))


def get_pilot_sample(seed: int = 42) -> list[TruthfulQATask]:
    """Get a pilot sample of 30 questions for initial testing.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of 30 stratified TruthfulQATask objects
    """
    loader = TruthfulQALoader()
    return loader.sample_stratified(n=30, seed=seed)


def get_main_sample(seed: int = 42) -> list[TruthfulQATask]:
    """Get main study sample of 150 questions.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of 150 stratified TruthfulQATask objects
    """
    loader = TruthfulQALoader()
    return loader.sample_stratified(n=150, seed=seed)


if __name__ == "__main__":
    # Test the loader
    print("Loading TruthfulQA dataset...")
    loader = TruthfulQALoader()
    dataset = loader.load()

    print(f"\nTotal questions: {len(dataset)}")
    print(f"\nCategories ({len(loader.get_categories())}): {loader.get_categories()}")

    # Show sample question
    sample = dataset[0]
    print("\nSample question:")
    print(f"  ID: {sample.id}")
    print(f"  Category: {sample.category}")
    print(f"  Question: {sample.question}")
    print(f"  Best answer: {sample.best_answer}")
    print(
        f"  Correct answers ({len(sample.correct_answers)}): {sample.correct_answers[:2]}..."
    )
    print(
        f"  Incorrect answers ({len(sample.incorrect_answers)}): {sample.incorrect_answers[:2]}..."
    )

    # Test stratified sampling
    print("\nTesting stratified sampling...")
    pilot = get_pilot_sample(seed=42)
    print(f"  Pilot sample size: {len(pilot)}")

    # Count categories in sample
    cat_counts: dict[str, int] = {}
    for task in pilot:
        cat_counts[task.category] = cat_counts.get(task.category, 0) + 1

    print(f"  Categories in pilot: {len(cat_counts)}")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"    {cat}: {count}")
