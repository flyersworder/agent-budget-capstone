# LLM-as-a-Judge Design Document

## Research Summary

Based on cutting-edge research (2024-2025), we've identified critical issues with our current heuristic evaluator and best practices for LLM-based evaluation.

### Key Research Findings

#### 1. Major Biases in LLM Judges
- **Position Bias**: Judges favor responses based on position (60-69% preference for "Response B")
- **Verbosity Bias**: Prefer longer, more formal outputs regardless of quality
- **Length Bias**: Favor longer responses even when conciseness is valuable
- **Self-Enhancement Bias**: Models favor their own outputs

#### 2. Validated Mitigation Strategies
- **Swapping and Averaging**: Swap response order and average scores (mitigates position bias)
- **Chain-of-Thought (CoT)**: Require reasoning before scoring (improves quality)
- **Detailed Rubrics**: Clear 1-5 scoring criteria for each dimension
- **Pairwise Comparison**: Compare responses head-to-head (more reliable than absolute scoring)
- **Abstract Labels**: Use "Response X" and "Response Y" instead of "Response 1/2"
- **Dimension Separation**: Evaluate each aspect separately (accuracy, clarity, etc.)
- **Reference-Guided**: Provide task expectations as context

#### 3. G-Eval Framework
The G-Eval framework (2024) provides a structured approach:
1. **Task Prompt Construction**: Define evaluation criteria clearly
2. **Chain-of-Thought Generation**: LLM generates reasoning steps
3. **Form-Filling Scoring**: LLM provides structured scores (1-5)
4. **Score Refinement**: Use probability-based aggregation

### Our Current Problems

**Current Evaluator (`experiments/evaluator.py`)**:
```python
# PROBLEM 1: Length bias built-in
if word_count <= max_len:
    length_score = 10.0  # Longer = better

# PROBLEM 2: No factual accuracy
# Only checks for keywords, not correctness

# PROBLEM 3: Favors verbose responses
# "Deep thinker" (concise but insightful) scores lower than "verbose" (long explanations)

# PROBLEM 4: No chain-of-thought
# Simple heuristics without reasoning
```

**Why This is Critical for Our Research**:
- We're comparing strategies where **concise ≠ bad** (deep thinker should score high with fewer words)
- Quality should be measured by **insight**, not **length**
- Current evaluator systematically favors verbose strategy

## Proposed LLM-Based Evaluator Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   LLMResponseEvaluator                   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  1. Pairwise Comparison (Position Bias Mitigation)      │
│     - Compare Strategy A vs B, B vs C, A vs C           │
│     - Swap order and average results                     │
│     - Use abstract labels (Response X, Response Y)       │
│                                                           │
│  2. Multi-Dimensional Rubric Evaluation                  │
│     - Accuracy: Factual correctness (1-5)               │
│     - Completeness: Addresses all aspects (1-5)         │
│     - Clarity: Clear communication (1-5)                │
│     - Depth: Analytical insight (1-5)                   │
│     - Conciseness: Efficiency of expression (1-5)      │
│                                                           │
│  3. Chain-of-Thought Reasoning                           │
│     - LLM explains reasoning for each dimension         │
│     - Provides evidence from responses                   │
│     - Justifies score before assigning it               │
│                                                           │
│  4. Aggregation and Calibration                          │
│     - Average across swapped positions                   │
│     - Combine pairwise results into rankings            │
│     - Output final scores with confidence intervals     │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Evaluation Dimensions

#### 1. **Accuracy** (1-5)
**Definition**: Factual correctness and reliability of information
- **5**: All facts correct, well-supported claims
- **4**: Mostly accurate with minor errors
- **3**: Generally accurate but some questionable claims
- **2**: Multiple factual errors or unsupported claims
- **1**: Significantly inaccurate or misleading

#### 2. **Completeness** (1-5)
**Definition**: Addresses all aspects of the question
- **5**: Fully addresses all parts of the question
- **4**: Covers most aspects, minor gaps
- **3**: Addresses main points but missing some aspects
- **2**: Incomplete, major aspects missing
- **1**: Barely addresses the question

#### 3. **Clarity** (1-5)
**Definition**: Clear, well-organized communication
- **5**: Exceptionally clear, well-structured
- **4**: Clear and organized
- **3**: Understandable but could be clearer
- **2**: Somewhat confusing or poorly organized
- **1**: Unclear or difficult to understand

#### 4. **Depth** (1-5)
**Definition**: Analytical insight and thoughtful analysis
- **5**: Deep insights, nuanced analysis, considers multiple perspectives
- **4**: Good analysis with meaningful insights
- **3**: Adequate analysis, surface-level insights
- **2**: Shallow analysis, lacks depth
- **1**: Superficial or no real analysis

#### 5. **Conciseness** (1-5)
**Definition**: Efficiency of expression (information per word)
- **5**: Maximally efficient, every word adds value
- **4**: Efficient, minimal redundancy
- **3**: Acceptable efficiency, some redundancy
- **2**: Verbose, significant redundancy
- **1**: Extremely verbose, poor information density

**CRITICAL**: Conciseness is **information density**, NOT brevity alone. A short but vague response scores low. A thorough but efficient response scores high.

### Prompt Templates

#### Template 1: Pairwise Comparison with Chain-of-Thought

```python
PAIRWISE_PROMPT = """You are evaluating two responses to a research question.

**Research Question**: {question}
**Task Complexity**: {complexity}
**Expected Coverage**: {expected_aspects}

**Response X**:
{response_x}

**Response Y**:
{response_y}

Evaluate these responses across five dimensions using the following rubrics:

**1. Accuracy (1-5)**: Factual correctness
- 5: All facts correct, well-supported
- 4: Mostly accurate, minor errors
- 3: Generally accurate, some questionable claims
- 2: Multiple errors or unsupported claims
- 1: Significantly inaccurate or misleading

**2. Completeness (1-5)**: Addresses all question aspects
- 5: Fully addresses all parts
- 4: Covers most aspects, minor gaps
- 3: Main points covered, some missing
- 2: Incomplete, major gaps
- 1: Barely addresses question

**3. Clarity (1-5)**: Clear, well-organized communication
- 5: Exceptionally clear and structured
- 4: Clear and organized
- 3: Understandable but could be clearer
- 2: Confusing or poorly organized
- 1: Unclear or difficult to understand

**4. Depth (1-5)**: Analytical insight and thoughtful analysis
- 5: Deep insights, nuanced, multiple perspectives
- 4: Good analysis, meaningful insights
- 3: Adequate, surface-level insights
- 2: Shallow, lacks depth
- 1: Superficial, no real analysis

**5. Conciseness (1-5)**: Information density (not brevity alone)
- 5: Maximally efficient, every word adds value
- 4: Efficient, minimal redundancy
- 3: Acceptable, some redundancy
- 2: Verbose, significant redundancy
- 1: Extremely verbose, poor information density

**Instructions**:
1. For each dimension, provide:
   a) Your reasoning (2-3 sentences analyzing both responses)
   b) Score for Response X (1-5)
   c) Score for Response Y (1-5)
   d) Which response is better (X, Y, or Tie)

2. Output format (JSON):
{{
  "accuracy": {{
    "reasoning": "...",
    "response_x_score": 4,
    "response_y_score": 5,
    "winner": "Y"
  }},
  "completeness": {{ ... }},
  "clarity": {{ ... }},
  "depth": {{ ... }},
  "conciseness": {{ ... }},
  "overall_winner": "Y"  // Based on which response wins more dimensions
}}

Focus on substance over style. A concise, insightful response can score higher than a lengthy, detailed one if it delivers more value per word.
"""
```

### Implementation Strategy

#### Phase 1: Core LLM Evaluator
```python
class LLMResponseEvaluator:
    def __init__(self, model="gemini-2.5-flash-lite"):
        self.model = model

    def evaluate_pairwise(
        self,
        response_x: str,
        response_y: str,
        task: ResearchTask
    ) -> PairwiseResult:
        """Compare two responses with position bias mitigation."""
        # Evaluate X vs Y
        result_xy = self._single_pairwise_evaluation(
            response_x, response_y, task
        )

        # Evaluate Y vs X (swapped)
        result_yx = self._single_pairwise_evaluation(
            response_y, response_x, task
        )

        # Average scores accounting for position swap
        return self._aggregate_swapped_results(result_xy, result_yx)
```

#### Phase 2: Position Bias Mitigation
- Swap response order
- Use abstract labels ("Response X", "Response Y")
- Average scores from both orderings
- Flag high disagreement between orderings (uncertainty indicator)

#### Phase 3: Strategy Ranking
```python
def rank_strategies(
    self,
    responses: dict[str, str],  # {strategy: response}
    task: ResearchTask
) -> StrategyRanking:
    """Rank all strategies using round-robin pairwise comparison."""

    # Compare all pairs
    pairwise_results = {}
    for strategy_a, strategy_b in itertools.combinations(responses.keys(), 2):
        comparison = self.evaluate_pairwise(
            responses[strategy_a],
            responses[strategy_b],
            task
        )
        pairwise_results[(strategy_a, strategy_b)] = comparison

    # Aggregate into rankings using Elo-style or win-count approach
    return self._aggregate_pairwise_rankings(pairwise_results)
```

### Validation Plan

#### 1. **Sanity Checks**
Create test cases with clear winners:
- Accurate vs. inaccurate response
- Complete vs. incomplete response
- Clear vs. confusing response

#### 2. **Position Bias Testing**
- Evaluate same pair twice (X vs Y, Y vs X)
- Measure score variance
- Target: < 0.5 point difference on 1-5 scale

#### 3. **Calibration with Human Judgment**
- Evaluate 10-20 responses manually
- Compare LLM scores to human scores
- Target: Correlation > 0.7

### Implementation Checklist

- [ ] Create `experiments/llm_evaluator.py` with LLMResponseEvaluator class
- [ ] Implement pairwise comparison with CoT prompting
- [ ] Add position bias mitigation (swapping + averaging)
- [ ] Design detailed rubrics for all 5 dimensions
- [ ] Implement round-robin ranking for 3 strategies
- [ ] Add validation suite with sanity checks
- [ ] Test position bias variance
- [ ] Document all prompts and design decisions
- [ ] Update experiment runner to use new evaluator
- [ ] Compare old vs. new evaluator on test data

### Expected Benefits

1. **Eliminates Length Bias**: Conciseness is explicitly valued
2. **Measures Quality**: Focuses on accuracy and insight, not word count
3. **Fair Comparison**: Position bias mitigation ensures fair head-to-head comparisons
4. **Interpretable**: Chain-of-thought provides reasoning for scores
5. **Research-Valid**: Aligns with cutting-edge best practices (2024-2025)

### Cost Considerations

**Current Heuristic Evaluator**: $0 per evaluation
**LLM Evaluator (gemini-2.5-flash)**: ~$0.002-0.02 per pairwise comparison

For 9 tasks × 3 pairwise comparisons × 2 positions (swap mitigation):
- Total: 54 LLM calls
- Estimated cost: $0.10-0.50 per experiment run

**Model Choice**: gemini-2.5-flash (not lite)
- **Rationale**: Evaluation quality is critical to research validity
- **Trade-off**: 2-3x more expensive than lite, but still negligible absolute cost
- **Benefit**: Better reasoning, more consistent judgments, fewer errors

**Recommendation**: Quality matters more than cost for evaluation. Use stronger model.

## References

1. "A Survey on LLM-as-a-Judge" (arXiv:2411.15594, 2024)
2. "LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods" (arXiv:2412.05579, 2024)
3. "Judging the Judges: Position Bias in Pairwise Comparative Assessments" (arXiv:2406.07791, 2024)
4. "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment" (2024)
5. "Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge" (2024)
6. "Evaluating and Mitigating LLM-as-a-judge Bias" (arXiv:2510.12462, 2024)
