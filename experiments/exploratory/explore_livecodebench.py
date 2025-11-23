"""Explore LiveCodeBench dataset to find suitable problems."""

from datasets import load_dataset


def main() -> None:
    print("Loading LiveCodeBench dataset...")

    # Load using the same approach as LiveCodeBench repo
    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag="release_v5",
        trust_remote_code=True,
    )

    print("Successfully loaded dataset!")

    print(f"Total problems: {len(dataset)}\n")

    # Show dataset structure
    print("Dataset features:")
    print(dataset.features)
    print()

    # Count by difficulty
    difficulty_counts: dict[str, int] = {}
    for item in dataset:
        diff = item.get("difficulty", "Unknown")
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

    print("Problems by difficulty:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"  {diff}: {count}")
    print()

    # Show a few examples from each difficulty
    print("=" * 80)
    print("SAMPLE PROBLEMS")
    print("=" * 80)

    for difficulty in ["easy", "medium", "hard"]:
        examples = [
            item for item in dataset if item.get("difficulty", "").lower() == difficulty
        ]
        if examples:
            print(f"\n{difficulty.upper()} Example:")
            print("-" * 80)
            ex = examples[0]
            print(f"Title: {ex.get('question_title', 'N/A')}")
            print(f"Platform: {ex.get('platform', 'N/A')}")
            print(f"Question ID: {ex.get('question_id', 'N/A')}")
            print("\nContent preview (first 500 chars):")
            content = ex.get("question_content", "")
            print(content[:500] + "..." if len(content) > 500 else content)
            print("\nStarter code preview (first 300 chars):")
            starter = ex.get("starter_code", "")
            print(starter[:300] + "..." if len(starter) > 300 else starter)
            print()


if __name__ == "__main__":
    main()
