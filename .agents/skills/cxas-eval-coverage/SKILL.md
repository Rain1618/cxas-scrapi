---
name: cxas-eval-coverage
description: "Calculates and generates design-time and execution coverage reports for Gemini Enterprise for Customer Experience (GECX) conversational agents by mapping evaluations against tools, callbacks and instructions."
---

# CXAS Evaluation Coverage Analyzer Skill

Use this skill to assess how comprehensively existing evaluations cover the agent's capabilities, specifically its tools, callbacks and instructions.

## Core Workflow Steps

1.  **Define Workspace Paths**:
    Identify the location of:
    *   The agent project root folder
    *   The tools folder (`tools/`)
    *   The evaluations folder:
        * `evaluations/` (contains Golden evals in separate folders)
        * `evaluationDatasets/` (contains shared eval datasets)
    *   The output directory for the coverage report, if there is no folder named `coverage_reports`, then create one at the root of the agent directory and output the coverage report there.

2.  **Run the Coverage Analysis Script**:
    Execute the `calculate_coverage.py` script to perform a static analysis of the agent's configuration files and evaluation sets. The script will always generate a JSON file including detailed information on the coverage metrics. Use `--output-file` to specify the JSON file path.

3.  **Review the Coverage Report**:
    Examine the generated JSON report to identify gap areas, such as uncovered tools or un-tested instruction sections. Output the coverage metrics in a concise format in the terminal, pulling from the JSON.

4.  **Generate Markdown Report (Optional)**:
    If the user explicitly asks for a detailed markdown report, pass the `--markdown-report /path/to/coverage_report.md` flag to `calculate_coverage.py` to generate it alongside the JSON report.
## Automation Scripts

### Calculate Coverage
`scripts/calculate_coverage.py`

Computes evaluation coverage percentages and generates a comprehensive report.

Usage:
```bash
python .agents/skills/cxas-eval-coverage/scripts/calculate_coverage.py \
  --agent-dir /path/to/agent/project \
  --output-file /path/to/coverage_report.json \
  --model gemini-2.5-flash \
  --markdown-report /path/to/coverage_report.md
```
*Note: The `--model` flag allows you to choose the Gemini model (default is `gemini-2.5-flash`, but `gemini-2.5-pro` can be used for higher reasoning accuracy).*

Supported Coverage Metrics:
*   **Tool Coverage**: Scans the `tools/` directory and maps each tool against explicit calls in Golden Evals or expectation strings in Simulation Evals.
*   **Callback Coverage**: Checks for unit tests associated with each callback.
*   **Instruction Segment Coverage**: Uses an XML tag fallback structure combined with an **LLM categorization pass** to filter out non-testable conversational fillers (maintaining line-by-line traceability) before performing vector-similarity-driven coverage analysis.
