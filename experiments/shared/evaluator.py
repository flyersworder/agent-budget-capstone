"""LLM-based response quality evaluator implementing best practices.

This module provides a research-grade LLM-as-a-Judge evaluator that:
- Uses pairwise comparison instead of absolute scoring
- Implements position bias mitigation via swapping and averaging
- Uses chain-of-thought reasoning with detailed rubrics
- Evaluates 5 dimensions: accuracy, completeness, clarity, depth, conciseness
- Provides interpretable reasoning for all scores

Based on cutting-edge research:
- G-Eval Framework (2024)
- Position Bias Mitigation (arXiv:2406.07791)
- LLM-as-a-Judge Survey (arXiv:2411.15594)
"""

import json
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from experiments.tasks import ResearchTask

# Load environment variables
load_dotenv()


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension.

    Attributes:
        reasoning: Chain-of-thought explanation for the score
        response_x_score: Score for first response (1-5)
        response_y_score: Score for second response (1-5)
        winner: Which response is better ("X", "Y", or "Tie")
    """

    reasoning: str
    response_x_score: float
    response_y_score: float
    winner: str


@dataclass
class PairwiseResult:
    """Result of comparing two responses.

    Attributes:
        accuracy: Factual correctness dimension
        completeness: Question coverage dimension
        clarity: Communication quality dimension
        depth: Analytical insight dimension
        conciseness: Information density dimension
        overall_winner: Overall best response ("X", "Y", or "Tie")
        confidence: Confidence in evaluation (low/medium/high)
    """

    accuracy: DimensionScore
    completeness: DimensionScore
    clarity: DimensionScore
    depth: DimensionScore
    conciseness: DimensionScore
    overall_winner: str
    confidence: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "accuracy": {
                "reasoning": self.accuracy.reasoning,
                "response_x_score": self.accuracy.response_x_score,
                "response_y_score": self.accuracy.response_y_score,
                "winner": self.accuracy.winner,
            },
            "completeness": {
                "reasoning": self.completeness.reasoning,
                "response_x_score": self.completeness.response_x_score,
                "response_y_score": self.completeness.response_y_score,
                "winner": self.completeness.winner,
            },
            "clarity": {
                "reasoning": self.clarity.reasoning,
                "response_x_score": self.clarity.response_x_score,
                "response_y_score": self.clarity.response_y_score,
                "winner": self.clarity.winner,
            },
            "depth": {
                "reasoning": self.depth.reasoning,
                "response_x_score": self.depth.response_x_score,
                "response_y_score": self.depth.response_y_score,
                "winner": self.depth.winner,
            },
            "conciseness": {
                "reasoning": self.conciseness.reasoning,
                "response_x_score": self.conciseness.response_x_score,
                "response_y_score": self.conciseness.response_y_score,
                "winner": self.conciseness.winner,
            },
            "overall_winner": self.overall_winner,
            "confidence": self.confidence,
        }


@dataclass
class AggregatedScore:
    """Aggregated score accounting for position bias.

    Attributes:
        dimension: Name of dimension
        score: Final score (1-5)
        confidence: Confidence level based on position consistency
        reasoning: Combined reasoning from both positions
    """

    dimension: str
    score: float
    confidence: str
    reasoning: str


@dataclass
class StrategyScore:
    """Complete score for a strategy.

    Attributes:
        strategy: Strategy name
        accuracy: Aggregated accuracy score
        completeness: Aggregated completeness score
        clarity: Aggregated clarity score
        depth: Aggregated depth score
        conciseness: Aggregated conciseness score
        overall: Overall score (average of dimensions)
    """

    strategy: str
    accuracy: float
    completeness: float
    clarity: float
    depth: float
    conciseness: float

    @property
    def overall(self) -> float:
        """Calculate overall score as average of all dimensions."""
        return (
            self.accuracy
            + self.completeness
            + self.clarity
            + self.depth
            + self.conciseness
        ) / 5

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary format."""
        return {
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "depth": self.depth,
            "conciseness": self.conciseness,
            "overall": self.overall,
        }


# Pairwise comparison prompt template with chain-of-thought and detailed rubrics
PAIRWISE_PROMPT = """You are evaluating two responses to a research question. Use chain-of-thought reasoning and detailed rubrics to assess quality.

**Research Question**: {question}
**Task Complexity**: {complexity}

**Response X**:
{response_x}

**Response Y**:
{response_y}

Evaluate these responses across five dimensions using the following rubrics:

**1. Accuracy (1-5)**: Factual correctness and reliability
- 5: All facts correct, well-supported claims, no misleading information
- 4: Mostly accurate with only minor errors that don't affect core message
- 3: Generally accurate but some questionable claims or unsupported assertions
- 2: Multiple factual errors or significant unsupported claims
- 1: Significantly inaccurate or misleading information

**2. Completeness (1-5)**: Addresses all aspects of the question
- 5: Fully addresses all parts of the question with appropriate depth
- 4: Covers most aspects with only minor gaps
- 3: Addresses main points but missing some important aspects
- 2: Incomplete coverage with major aspects missing
- 1: Barely addresses the question or misses core requirements

**3. Clarity (1-5)**: Clear, well-organized communication
- 5: Exceptionally clear, well-structured, easy to follow
- 4: Clear and organized with good flow
- 3: Understandable but could be clearer or better organized
- 2: Somewhat confusing, poorly organized, or hard to follow
- 1: Unclear, disorganized, or very difficult to understand

**4. Depth (1-5)**: Analytical insight and thoughtful analysis
- 5: Deep insights, nuanced analysis, considers multiple perspectives
- 4: Good analysis with meaningful insights and context
- 3: Adequate analysis with surface-level insights
- 2: Shallow analysis that lacks meaningful depth
- 1: Superficial or no real analysis

**5. Conciseness (1-5)**: Information density (value per word)
- 5: Maximally efficient, every word adds value, no redundancy
- 4: Efficient communication with minimal redundancy
- 3: Acceptable efficiency with some redundancy or filler
- 2: Verbose with significant redundancy or unnecessary detail
- 1: Extremely verbose with poor information density

**IMPORTANT**: Conciseness measures information density, NOT brevity alone. A short but vague response scores low. A thorough but efficient response scores high.

**Instructions**:
For each dimension:
1. Think step-by-step about both responses
2. Provide your reasoning (2-3 sentences analyzing both responses)
3. Assign scores (1-5) based on the rubric
4. Determine the winner (X, Y, or Tie)

Output your evaluation as a JSON object with this exact structure:
{{
  "accuracy": {{
    "reasoning": "Your analysis here",
    "response_x_score": 4,
    "response_y_score": 5,
    "winner": "Y"
  }},
  "completeness": {{
    "reasoning": "Your analysis here",
    "response_x_score": 4,
    "response_y_score": 4,
    "winner": "Tie"
  }},
  "clarity": {{
    "reasoning": "Your analysis here",
    "response_x_score": 5,
    "response_y_score": 4,
    "winner": "X"
  }},
  "depth": {{
    "reasoning": "Your analysis here",
    "response_x_score": 4,
    "response_y_score": 5,
    "winner": "Y"
  }},
  "conciseness": {{
    "reasoning": "Your analysis here",
    "response_x_score": 5,
    "response_y_score": 3,
    "winner": "X"
  }},
  "overall_winner": "Y"
}}

**Overall winner** should be the response that wins the most dimensions (or "Tie" if equal).
Focus on substance over style. Think carefully about each dimension.
"""


class LLMResponseEvaluator:
    """LLM-based evaluator with position bias mitigation.

    Uses Google's Gemini model to evaluate responses with:
    - Pairwise comparison (more reliable than absolute scoring)
    - Position bias mitigation (swapping and averaging)
    - Chain-of-thought reasoning
    - Detailed rubrics for 5 dimensions

    Default model: gemini-2.5-flash (prioritizes quality over cost for evaluation)
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        """Initialize the evaluator.

        Args:
            model: Gemini model to use for evaluation (default: gemini-2.5-flash for quality)
        """
        self.model = model

        # Initialize Gemini client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment. "
                "Please set it in .env file or environment variables."
            )

        self.client = genai.Client(api_key=api_key)

    def _call_llm_judge(self, prompt: str) -> dict[str, Any]:
        """Call LLM to evaluate responses.

        Args:
            prompt: Evaluation prompt

        Returns:
            Parsed JSON evaluation result
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,  # Deterministic for consistency
                response_mime_type="application/json",  # Request JSON output
            ),
        )

        # Parse JSON response
        try:
            result: dict[str, Any] = json.loads(response.text)
            return result
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response: {response.text}")
            raise ValueError(f"LLM did not return valid JSON: {e}")

    def _single_pairwise_evaluation(
        self, response_x: str, response_y: str, task: ResearchTask
    ) -> PairwiseResult:
        """Perform a single pairwise evaluation (one position).

        Args:
            response_x: First response
            response_y: Second response
            task: Research task

        Returns:
            PairwiseResult with scores for all dimensions
        """
        # Create prompt
        prompt = PAIRWISE_PROMPT.format(
            question=task.question,
            complexity=task.complexity,
            response_x=response_x,
            response_y=response_y,
        )

        # Get LLM evaluation
        result = self._call_llm_judge(prompt)

        # Parse results into structured format
        return PairwiseResult(
            accuracy=DimensionScore(
                reasoning=result["accuracy"]["reasoning"],
                response_x_score=result["accuracy"]["response_x_score"],
                response_y_score=result["accuracy"]["response_y_score"],
                winner=result["accuracy"]["winner"],
            ),
            completeness=DimensionScore(
                reasoning=result["completeness"]["reasoning"],
                response_x_score=result["completeness"]["response_x_score"],
                response_y_score=result["completeness"]["response_y_score"],
                winner=result["completeness"]["winner"],
            ),
            clarity=DimensionScore(
                reasoning=result["clarity"]["reasoning"],
                response_x_score=result["clarity"]["response_x_score"],
                response_y_score=result["clarity"]["response_y_score"],
                winner=result["clarity"]["winner"],
            ),
            depth=DimensionScore(
                reasoning=result["depth"]["reasoning"],
                response_x_score=result["depth"]["response_x_score"],
                response_y_score=result["depth"]["response_y_score"],
                winner=result["depth"]["winner"],
            ),
            conciseness=DimensionScore(
                reasoning=result["conciseness"]["reasoning"],
                response_x_score=result["conciseness"]["response_x_score"],
                response_y_score=result["conciseness"]["response_y_score"],
                winner=result["conciseness"]["winner"],
            ),
            overall_winner=result["overall_winner"],
        )

    def _aggregate_swapped_results(
        self,
        result_xy: PairwiseResult,
        result_yx: PairwiseResult,
        strategy_x: str,
        strategy_y: str,
    ) -> tuple[dict[str, AggregatedScore], dict[str, AggregatedScore]]:
        """Aggregate results from both position orderings.

        Mitigates position bias by averaging scores from swapped positions.

        Args:
            result_xy: Result with X first, Y second
            result_yx: Result with Y first, X second (swapped)
            strategy_x: Name of strategy X
            strategy_y: Name of strategy Y

        Returns:
            Tuple of (aggregated_score_x, aggregated_score_y) for each dimension
        """
        aggregated_x = {}
        aggregated_y = {}

        dimensions = ["accuracy", "completeness", "clarity", "depth", "conciseness"]

        for dim in dimensions:
            # Get dimension scores from both orderings
            dim_xy = getattr(result_xy, dim)
            dim_yx = getattr(result_yx, dim)

            # X's score: average of (X's score when X is first) and (X's score when X is second)
            # When X is first: use response_x_score from result_xy
            # When X is second (Y is first): use response_y_score from result_yx
            score_x = (dim_xy.response_x_score + dim_yx.response_y_score) / 2

            # Y's score: average of (Y's score when Y is second) and (Y's score when Y is first)
            score_y = (dim_xy.response_y_score + dim_yx.response_x_score) / 2

            # Calculate confidence based on score consistency
            score_diff = abs(
                (dim_xy.response_x_score - dim_xy.response_y_score)
                - (dim_yx.response_y_score - dim_yx.response_x_score)
            )

            if score_diff <= 0.5:
                confidence = "high"
            elif score_diff <= 1.5:
                confidence = "medium"
            else:
                confidence = "low"

            # Combine reasoning
            combined_reasoning = (
                f"[X first] {dim_xy.reasoning} | [Y first] {dim_yx.reasoning}"
            )

            aggregated_x[dim] = AggregatedScore(
                dimension=dim,
                score=score_x,
                confidence=confidence,
                reasoning=combined_reasoning,
            )

            aggregated_y[dim] = AggregatedScore(
                dimension=dim,
                score=score_y,
                confidence=confidence,
                reasoning=combined_reasoning,
            )

        return aggregated_x, aggregated_y

    def evaluate_pairwise_with_swap(
        self,
        response_x: str,
        response_y: str,
        task: ResearchTask,
        label_x: str,
        label_y: str,
    ) -> tuple[dict[str, AggregatedScore], dict[str, AggregatedScore]]:
        """Compare two responses with position bias mitigation.

        Evaluates responses in both orders and averages results.

        Args:
            response_x: First response
            response_y: Second response
            task: Research task
            label_x: Label for first strategy (e.g., "deep")
            label_y: Label for second strategy (e.g., "balanced")

        Returns:
            Tuple of (scores_x, scores_y) where each is a dict mapping dimension to AggregatedScore
        """
        print(f"  Evaluating {label_x} vs {label_y} (position 1)...")
        result_xy = self._single_pairwise_evaluation(response_x, response_y, task)

        print(f"  Evaluating {label_y} vs {label_x} (position 2, swapped)...")
        result_yx = self._single_pairwise_evaluation(response_y, response_x, task)

        # Aggregate scores accounting for position swap
        scores_x, scores_y = self._aggregate_swapped_results(
            result_xy, result_yx, label_x, label_y
        )

        return scores_x, scores_y

    def rank_strategies(
        self, responses: dict[str, str], task: ResearchTask
    ) -> dict[str, StrategyScore]:
        """Rank all strategies using round-robin pairwise comparison.

        Args:
            responses: Dictionary mapping strategy name to response text
            task: Research task

        Returns:
            Dictionary mapping strategy name to StrategyScore
        """
        strategies = list(responses.keys())

        # Initialize cumulative scores
        cumulative_scores: dict[str, dict[str, list[float]]] = {
            strategy: {
                "accuracy": [],
                "completeness": [],
                "clarity": [],
                "depth": [],
                "conciseness": [],
            }
            for strategy in strategies
        }

        # Perform round-robin pairwise comparisons
        print(f"\nEvaluating task: {task.id}")
        print(f"Question: {task.question}")

        # Compare all pairs
        import itertools

        for strategy_a, strategy_b in itertools.combinations(strategies, 2):
            print(f"\n--- Comparing {strategy_a} vs {strategy_b} ---")

            scores_a, scores_b = self.evaluate_pairwise_with_swap(
                responses[strategy_a],
                responses[strategy_b],
                task,
                strategy_a,
                strategy_b,
            )

            # Accumulate scores for averaging
            for dim in ["accuracy", "completeness", "clarity", "depth", "conciseness"]:
                cumulative_scores[strategy_a][dim].append(scores_a[dim].score)
                cumulative_scores[strategy_b][dim].append(scores_b[dim].score)

        # Average scores across all pairwise comparisons
        final_scores = {}
        for strategy in strategies:
            final_scores[strategy] = StrategyScore(
                strategy=strategy,
                accuracy=sum(cumulative_scores[strategy]["accuracy"])
                / len(cumulative_scores[strategy]["accuracy"]),
                completeness=sum(cumulative_scores[strategy]["completeness"])
                / len(cumulative_scores[strategy]["completeness"]),
                clarity=sum(cumulative_scores[strategy]["clarity"])
                / len(cumulative_scores[strategy]["clarity"]),
                depth=sum(cumulative_scores[strategy]["depth"])
                / len(cumulative_scores[strategy]["depth"]),
                conciseness=sum(cumulative_scores[strategy]["conciseness"])
                / len(cumulative_scores[strategy]["conciseness"]),
            )

        return final_scores

    def evaluate_response(self, response: str, task: ResearchTask) -> dict[str, float]:
        """Evaluate a single response (legacy interface for compatibility).

        This method provides backward compatibility with the old evaluator interface.
        For new code, use rank_strategies() instead.

        Args:
            response: Response text
            task: Research task

        Returns:
            Dictionary with scores (for compatibility, all set to 0)
        """
        # Legacy interface - return dummy scores
        # New code should use rank_strategies() instead
        return {
            "completeness": 0.0,
            "clarity": 0.0,
            "depth": 0.0,
            "evidence": 0.0,
            "overall": 0.0,
        }


# Backward compatibility alias
ResponseEvaluator = LLMResponseEvaluator
