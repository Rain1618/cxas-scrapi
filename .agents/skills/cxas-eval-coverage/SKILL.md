---
name: cxas-eval-coverage
description: "Calculates and generates design-time and execution coverage reports for Gemini Enterprise for Customer Experience (GECX) conversational agents by mapping evaluations (Goldens and Simulation Evals) against tools and instructions."
---

# CXAS Evaluation Coverage Analyzer Skill

Use this skill to assess how comprehensively existing evaluations cover the agent's capabilities, specifically its tools and instructions.

## Core Workflow Steps

1.  **Define Workspace Paths**:
    Identify the location of:
    *   The tools folder (`tools/`)
    *   The evaluations folder:
        * `evaluations/` (contains Golden evals in separate folders)
        * `evaluationDatasets/` (contains shared eval datasets)
    *   The agent instruction file (`instruction.txt`)
    *   The output directory for the coverage report, if there is no folder named `coverage_reports`, then create one at the root of the agent directory and output the coverage report there

2.  **Run the Coverage Analysis Script**:
    Execute the `calculate_coverage.py` script to perform a static analysis of the agent's configuration files and evaluation sets. If the user asks for a detailed report, then generate the `coverage_report.md` file inside the coverage_reports folder. Otherwise, output the coverage metrics in a concise format in the terminal.

3.  **Review the Coverage Report**:
    Examine the generated markdown report (`coverage_report.md`) to identify gap areas, such as uncovered tools or un-tested instruction sections.

## Automation Scripts

### Calculate Coverage
`scripts/calculate_coverage.py`

Computes evaluation coverage percentages and generates a comprehensive report.

Usage:
```bash
python .agents/skills/cxas-eval-coverage/scripts/calculate_coverage.py \
  --agent-dir /path/to/agent/project \
  --output-file ./coverage_reports/coverage_report.md
```

Supported Coverage Metrics:
*   **Tool Coverage**: Scans the `tools/` directory and maps each tool against explicit calls in Golden Evals or expectation strings in Simulation Evals.
*   **Intent/Tag Coverage**: Maps the tags defined in evaluations against the intents/actions defined in the agent instruction.
*   **Instruction Segment Coverage**: Uses an XML tag fallback structure combined with a **Pro LLM-based consolidation pass** to merge redundant instructions and filter out non-testable conversational fillers before performing vector-similarity-driven coverage analysis.
