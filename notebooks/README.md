# Analysis Notebooks

This directory contains Jupyter notebooks for analyzing experiment results.

## Setup

Install analysis dependencies:
```bash
uv sync --group analysis
```

## Notebooks

### part1_analysis.ipynb
Comprehensive analysis of Part 1 experiments (Single-Agent Strategic Allocation):
- Token allocation patterns across strategies
- Quality metrics comparison
- Tool usage analysis
- Efficiency metrics
- Visualization of results

## Running the Notebooks

### Option 1: Jupyter Lab (Recommended)
```bash
uv run jupyter lab
```

### Option 2: Jupyter Notebook
```bash
uv run jupyter notebook
```

### Option 3: VS Code
Open the `.ipynb` file in VS Code with the Jupyter extension installed.

## Workflow

1. Run experiments to generate results:
   ```bash
   # Test run
   uv run python -m experiments.run_part1 --test

   # Full run
   uv run python -m experiments.run_part1
   ```

2. Open the corresponding notebook to analyze results

3. The notebook will automatically load the most recent results file

## Output

Notebooks generate:
- Visualizations saved to `experiments/results/*.png`
- Summary tables saved to `experiments/results/*.csv`
- Interactive analysis in the notebook
