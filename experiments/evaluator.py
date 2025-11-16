"""Response quality evaluator for experiment analysis."""

from dataclasses import dataclass
from typing import Any

from experiments.tasks import ResearchTask


@dataclass
class QualityScore:
    """Quality assessment for a response.

    Attributes:
        completeness: How fully the response addresses the question (0-10)
        clarity: How clear and well-structured the response is (0-10)
        depth: How detailed and thorough the analysis is (0-10)
        evidence: How well the response uses evidence/sources (0-10)
        overall: Overall quality score (average of components)
    """

    completeness: float
    clarity: float
    depth: float
    evidence: float

    @property
    def overall(self) -> float:
        """Calculate overall quality score.

        Returns:
            Average of all component scores
        """
        return (self.completeness + self.clarity + self.depth + self.evidence) / 4

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary format.

        Returns:
            Dictionary with all scores
        """
        return {
            "completeness": self.completeness,
            "clarity": self.clarity,
            "depth": self.depth,
            "evidence": self.evidence,
            "overall": self.overall,
        }


class ResponseEvaluator:
    """Evaluates response quality using heuristic-based analysis.

    This class provides automated quality assessment of agent responses
    based on structural and content analysis. For production use, consider
    LLM-based evaluation for more nuanced quality assessment.
    """

    def evaluate_response(self, response: str, task: ResearchTask) -> QualityScore:
        """Evaluate a response against a research task.

        Args:
            response: The agent's response text
            task: The research task that was executed

        Returns:
            QualityScore with component and overall scores
        """
        completeness = self._evaluate_completeness(response, task)
        clarity = self._evaluate_clarity(response)
        depth = self._evaluate_depth(response, task)
        evidence = self._evaluate_evidence(response)

        return QualityScore(
            completeness=completeness, clarity=clarity, depth=depth, evidence=evidence
        )

    def _evaluate_completeness(self, response: str, task: ResearchTask) -> float:
        """Evaluate how completely the response addresses the question.

        Uses heuristics based on response length, question complexity,
        and keyword coverage.

        Args:
            response: Response text
            task: Research task

        Returns:
            Completeness score (0-10)
        """
        if not response:
            return 0.0

        # Base score on response length relative to expected length
        # Simple tasks: ~200-400 words
        # Moderate tasks: ~400-600 words
        # Complex tasks: ~600-800 words

        word_count = len(response.split())

        expected_lengths = {
            "simple": (200, 400),
            "moderate": (400, 600),
            "complex": (600, 800),
        }

        min_len, max_len = expected_lengths.get(task.complexity, (300, 500))

        # Score based on length appropriateness
        if word_count < min_len * 0.5:
            length_score = 3.0  # Too short
        elif word_count < min_len:
            length_score = 6.0  # Somewhat short
        elif word_count <= max_len:
            length_score = 10.0  # Appropriate length
        elif word_count <= max_len * 1.5:
            length_score = 8.0  # Slightly long
        else:
            length_score = 6.0  # Too long

        # Check for key question terms in response
        question_words = set(task.question.lower().split())
        response_words = set(response.lower().split())

        # Filter out common words
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "what",
            "how",
            "why",
            "when",
            "where",
            "which",
            "who",
        }
        question_keywords = question_words - stopwords

        if question_keywords:
            keyword_coverage = len(question_keywords & response_words) / len(
                question_keywords
            )
            keyword_score = keyword_coverage * 10
        else:
            keyword_score = 5.0

        # Average length and keyword scores
        return (length_score + keyword_score) / 2

    def _evaluate_clarity(self, response: str) -> float:
        """Evaluate response clarity and structure.

        Uses heuristics based on sentence structure, paragraph organization,
        and readability indicators.

        Args:
            response: Response text

        Returns:
            Clarity score (0-10)
        """
        if not response:
            return 0.0

        # Check for paragraph structure
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        has_paragraphs = len(paragraphs) > 1
        paragraph_score = 10.0 if has_paragraphs else 6.0

        # Check for list/bullet points (indicates organization)
        has_lists = any(
            line.strip().startswith(("-", "*", "•", str(i) + "."))
            for line in response.split("\n")
            for i in range(1, 10)
        )
        list_score = 10.0 if has_lists else 7.0

        # Check sentence length variety (good writing has varied sentences)
        sentences = [
            s.strip()
            for s in response.replace("?", ".").replace("!", ".").split(".")
            if s.strip()
        ]
        if len(sentences) > 2:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(
                sentences
            )
            # Ideal average is 15-20 words per sentence
            if 12 <= avg_sentence_length <= 25:
                sentence_score = 10.0
            else:
                sentence_score = 7.0
        else:
            sentence_score = 5.0

        # Average all clarity indicators
        return (paragraph_score + list_score + sentence_score) / 3

    def _evaluate_depth(self, response: str, task: ResearchTask) -> float:
        """Evaluate depth of analysis and detail.

        Args:
            response: Response text
            task: Research task

        Returns:
            Depth score (0-10)
        """
        if not response:
            return 0.0

        # Look for indicators of deep analysis
        analysis_indicators = [
            "however",
            "furthermore",
            "additionally",
            "moreover",
            "in contrast",
            "on the other hand",
            "consequently",
            "therefore",
            "thus",
            "hence",
            "specifically",
            "for example",
            "for instance",
            "such as",
        ]

        indicator_count = sum(
            1 for indicator in analysis_indicators if indicator in response.lower()
        )

        # More indicators suggest deeper analysis
        indicator_score = min(indicator_count * 2, 10)

        # Check for technical terms (domain-specific vocabulary)
        # Longer words often indicate technical depth
        words = response.split()
        long_words = [w for w in words if len(w) > 10]
        technical_ratio = len(long_words) / len(words) if words else 0
        technical_score = min(technical_ratio * 50, 10)

        # Complexity-adjusted expectations
        complexity_multiplier = {"simple": 0.7, "moderate": 1.0, "complex": 1.2}.get(
            task.complexity, 1.0
        )

        base_score = (indicator_score + technical_score) / 2
        return min(base_score * complexity_multiplier, 10.0)

    def _evaluate_evidence(self, response: str) -> float:
        """Evaluate use of evidence and sources.

        Args:
            response: Response text

        Returns:
            Evidence score (0-10)
        """
        if not response:
            return 0.0

        # Look for evidence indicators
        evidence_patterns = [
            "according to",
            "research shows",
            "studies indicate",
            "data suggests",
            "evidence shows",
            "reported that",
            "found that",
            "demonstrated",
            "proven",
            "statistics",
        ]

        evidence_count = sum(
            1 for pattern in evidence_patterns if pattern in response.lower()
        )

        # Check for specific numbers/statistics
        import re

        numbers = re.findall(r"\b\d+\.?\d*\s*%|\b\d+\.?\d*\s*percent", response.lower())
        has_statistics = len(numbers) > 0

        # Check for citations or references
        has_citations = any(
            marker in response for marker in ["[", "(", "source:", "ref:"]
        )

        # Scoring
        evidence_score = min(evidence_count * 2.5, 6)
        if has_statistics:
            evidence_score += 2
        if has_citations:
            evidence_score += 2

        return min(evidence_score, 10.0)

    def compare_responses(
        self, responses: dict[str, str], task: ResearchTask
    ) -> dict[str, Any]:
        """Compare multiple responses for the same task.

        Args:
            responses: Dictionary mapping strategy name to response text
            task: The research task

        Returns:
            Comparison results with scores for each strategy
        """
        scores = {}
        for strategy, response in responses.items():
            scores[strategy] = self.evaluate_response(response, task).to_dict()

        # Find best performing strategy for each dimension
        best = {
            "completeness": max(scores.items(), key=lambda x: x[1]["completeness"])[0],
            "clarity": max(scores.items(), key=lambda x: x[1]["clarity"])[0],
            "depth": max(scores.items(), key=lambda x: x[1]["depth"])[0],
            "evidence": max(scores.items(), key=lambda x: x[1]["evidence"])[0],
            "overall": max(scores.items(), key=lambda x: x[1]["overall"])[0],
        }

        return {"task_id": task.id, "scores": scores, "best_performers": best}
