"""Objective correctness evaluator for HotpotQA responses.

This evaluator uses LLM-as-judge with objective ground truth to score
factual correctness on a binary/ternary scale for multi-hop questions.
"""

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from experiments.tasks.hotpotqa_tasks import HotpotQATask

# Load environment variables
load_dotenv()


@dataclass
class HotpotQAScore:
    """Correctness evaluation score for HotpotQA.

    Attributes:
        score: Correctness score (0.0 = incorrect, 0.5 = partial, 1.0 = correct)
        justification: Brief explanation of the score
        question: The question that was evaluated
        correct_answer: The ground truth answer
        agent_response: The agent's response
        question_type: Type of question ("bridge" or "comparison")
    """

    score: float
    justification: str
    question: str
    correct_answer: str
    agent_response: str
    question_type: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "score": self.score,
            "justification": self.justification,
            "question": self.question,
            "correct_answer": self.correct_answer,
            "agent_response": self.agent_response,
            "question_type": self.question_type,
        }


class HotpotQAEvaluator:
    """Evaluates factual correctness for HotpotQA using LLM-as-judge."""

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        """Initialize evaluator.

        Args:
            model: Gemini model to use for evaluation
        """
        self.model = model
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def evaluate_correctness(
        self, task: HotpotQATask, agent_response: str
    ) -> HotpotQAScore:
        """Evaluate factual correctness of agent response.

        Args:
            task: HotpotQA task with ground truth
            agent_response: Agent's response to evaluate

        Returns:
            HotpotQAScore with score and justification
        """
        # Create evaluation prompt
        prompt = self._create_evaluation_prompt(task, agent_response)

        # Get LLM evaluation
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,  # Deterministic for consistency
                max_output_tokens=500,
            ),
        )

        # Parse response
        score, justification = self._parse_evaluation(response.text)

        return HotpotQAScore(
            score=score,
            justification=justification,
            question=task.question,
            correct_answer=task.answer,
            agent_response=agent_response,
            question_type=task.question_type,
        )

    def _create_evaluation_prompt(self, task: HotpotQATask, agent_response: str) -> str:
        """Create evaluation prompt with ground truth.

        Args:
            task: HotpotQA task
            agent_response: Agent's response

        Returns:
            Evaluation prompt string
        """
        # Add context based on question type
        type_context = ""
        if task.question_type == "bridge":
            type_context = (
                "This is a bridge question requiring multi-hop reasoning by "
                "chaining multiple facts together."
            )
        elif task.question_type == "comparison":
            type_context = (
                "This is a comparison question requiring comparing two entities "
                "to determine if they share a property."
            )

        return f"""You are evaluating the factual correctness of an AI agent's response to a multi-hop question.

Question Type: {task.question_type}
{type_context}

Question: {task.question}

Ground Truth Answer: {task.answer}

Agent's Response: {agent_response}

Task: Evaluate if the agent's response is factually correct compared to the ground truth.

Scoring Guidelines:
- 1.0 (Correct): Response contains the correct answer. The agent may provide reasoning or additional context, but the core factual answer must match the ground truth.
- 0.5 (Partial): Response is on the right track but incomplete, ambiguous, or contains minor inaccuracies that don't completely invalidate the answer.
- 0.0 (Incorrect): Response provides the wrong answer, contradicts the ground truth, or fails to answer the question.

Important Considerations:
- For names: Accept variations in spelling or formatting (e.g., "JFK" = "John F. Kennedy")
- For yes/no questions: Accept "yes"/"no", "true"/"false", or equivalent affirmative/negative statements
- For numerical answers: Accept minor variations in precision if the core value is correct
- Focus on factual accuracy, not verbosity or reasoning quality
- "I don't know" scores 0.5 if the agent acknowledges uncertainty (better than confident wrong answer)
- If the agent provides reasoning that leads to the correct answer, score 1.0 even if verbose

Provide your evaluation in exactly this format:
SCORE: [0.0, 0.5, or 1.0]
JUSTIFICATION: [1-2 sentence explanation]"""

    def _parse_evaluation(self, response_text: str) -> tuple[float, str]:
        """Parse LLM evaluation response.

        Args:
            response_text: Raw LLM response

        Returns:
            Tuple of (score, justification)
        """
        lines = response_text.strip().split("\n")

        score = 0.0
        justification = ""

        for line in lines:
            line = line.strip()
            if line.startswith("SCORE:"):
                score_str = line.replace("SCORE:", "").strip()
                try:
                    score = float(score_str)
                    # Clamp to valid range
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    # Default to 0.0 if parse fails
                    score = 0.0
            elif line.startswith("JUSTIFICATION:"):
                justification = line.replace("JUSTIFICATION:", "").strip()

        return score, justification


if __name__ == "__main__":
    # Test the evaluator
    from experiments.tasks.hotpotqa_tasks import get_pilot_sample

    print("=" * 80)
    print("HotpotQA Correctness Evaluator Test")
    print("=" * 80)
    print()

    # Load sample tasks
    tasks = get_pilot_sample(seed=42)
    test_task = tasks[0]

    print(f"Test Question: {test_task.question}")
    print(f"Question Type: {test_task.question_type}")
    print(f"Correct Answer: {test_task.answer}")
    print()

    # Create evaluator
    evaluator = HotpotQAEvaluator()

    # Test with correct response
    print("Test 1: Correct Response")
    correct_response = f"The answer is {test_task.answer}."
    print(f"  Response: {correct_response}")
    score1 = evaluator.evaluate_correctness(test_task, correct_response)
    print(f"  Score: {score1.score}")
    print(f"  Justification: {score1.justification}")
    print()

    # Test with reasoning that leads to correct answer
    print("Test 2: Correct Response with Reasoning")
    reasoning_response = (
        f"Based on my research, I found that {test_task.answer} is the answer "
        f"because this satisfies the multi-hop reasoning required."
    )
    print(f"  Response: {reasoning_response}")
    score2 = evaluator.evaluate_correctness(test_task, reasoning_response)
    print(f"  Score: {score2.score}")
    print(f"  Justification: {score2.justification}")
    print()

    # Test with incorrect response
    print("Test 3: Incorrect Response")
    incorrect_response = "The answer is Paris."
    print(f"  Response: {incorrect_response}")
    score3 = evaluator.evaluate_correctness(test_task, incorrect_response)
    print(f"  Score: {score3.score}")
    print(f"  Justification: {score3.justification}")
    print()

    # Test with ambiguous response
    print("Test 4: Ambiguous/Uncertain Response")
    ambiguous_response = "I'm not sure, but I think it might be related to Europe."
    print(f"  Response: {ambiguous_response}")
    score4 = evaluator.evaluate_correctness(test_task, ambiguous_response)
    print(f"  Score: {score4.score}")
    print(f"  Justification: {score4.justification}")
    print()

    print("=" * 80)
    print("Evaluator test completed successfully!")
    print("=" * 80)
