#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main execution script for GECX evaluation coverage analyzer."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from ingestion import ingest_agent_project
from instruction_coverage import (
    analyze_instruction_categories,
    extract_instruction_coverage,
    determine_desired_transfers_with_llm,
)

from cxas_scrapi.utils.gemini import GeminiGenerate


def generate_report(
    output_file: Path,
    total_tools: Set[str],
    covered_tools: Set[str],
    phantom_tools_by_file: Dict[Path, Set[str]],
    eval_files: List[Path],
    declared_transfers: List[Tuple[str, str]],
    covered_transfers: Dict[Tuple[str, str], List[str]],
    instruction_segments: List[Dict[str, Any]],
    covered_instruction_segments: List[Dict[str, Any]],
    instruction_files: List[Path],
    agent_dir: Path,
    total_callbacks: Set[str],
    covered_callbacks: Set[str],
    desired_transfers: Set[Tuple[str, str]],
    errors: List[str] = None,
) -> None:
    """Generates a comprehensive coverage report."""
    uncovered_tools = total_tools - covered_tools
    tool_coverage_pct = (
        (len(covered_tools) / len(total_tools) * 100.0) if total_tools else 0.0
    )

    total_segments = len(instruction_segments)
    total_covered = len(covered_instruction_segments)
    overall_segment_pct = (
        (total_covered / total_segments * 100.0) if total_segments else 0.0
    )

    total_transfers = len(declared_transfers)
    total_transfers_covered = len(covered_transfers)
    transfer_coverage_pct = (
        (total_transfers_covered / total_transfers * 100.0)
        if total_transfers
        else 0.0
    )

    total_cbs = len(total_callbacks)
    covered_cbs = len(covered_callbacks)
    callback_coverage_pct = (
        (covered_cbs / total_cbs * 100.0) if total_cbs else 0.0
    )

    category_counts = {}
    category_covered_counts = {}

    for instruction_segment in instruction_segments:
        cat = instruction_segment["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        if instruction_segment["covered"] == "Yes":
            category_covered_counts[cat] = (
                category_covered_counts.get(cat, 0) + 1
            )

    report = []
    report.append("# Evaluation Coverage Report\n")

    if errors:
        report.append("> [!CAUTION]")
        report.append(
            "> **API Errors Occurred:** The coverage calculations may be "
            "inaccurate or incomplete due to the following errors during execution:"
        )
        for err in set(errors):
            report.append(f"> *   {err}")
        report.append("\n")

    if phantom_tools_by_file:
        report.append("> [!WARNING]")
        report.append(
            "> The following tools are referenced in evaluations but "
            "do not exist in the `tools/` directory:"
        )
        for ef, phantoms in sorted(phantom_tools_by_file.items()):
            phantoms_str = ", ".join(f"`{p}`" for p in sorted(phantoms))
            report.append(f"> *   `{ef.name}`: {phantoms_str}")
        report.append("\n")

    report.append("## Summary Metrics\n")
    report.append("| Metric | Total | Covered | Coverage % |")
    report.append("| :--- | :---: | :---: | :---: |")
    report.append(
        f"| **Tool Integrations** | {len(total_tools)} | "
        f"{len(covered_tools)} | {tool_coverage_pct:.1f}% |"
    )
    report.append(
        f"| **Instruction Segments** | {total_segments} | "
        f"{total_covered} | {overall_segment_pct:.1f}% |"
    )
    report.append(
        f"| **Agent Transfers** | {total_transfers} | "
        f"{total_transfers_covered} | {transfer_coverage_pct:.1f}% |"
    )
    report.append(
        f"| **Callbacks** | {total_cbs} | "
        f"{covered_cbs} | {callback_coverage_pct:.1f}% |"
    )
    report.append("\n")

    report.append("## Instruction Segment Category Breakdown\n")
    report.append("| Category | Total | Covered | Coverage % |")
    report.append("| :--- | :---: | :---: | :---: |")

    for cat in sorted(category_counts.keys()):
        total = category_counts[cat]
        covered = category_covered_counts.get(cat, 0)
        pct = (covered / total * 100.0) if total else 0.0
        report.append(f"| **{cat}** | {total} | {covered} | {pct:.1f}% |")

    report.append("\n---\n")

    report.append("## Uncovered Segments\n")
    has_uncovered = False
    for instruction_segment in instruction_segments:
        if instruction_segment["covered"] == "No":
            if not has_uncovered:
                report.append("### Uncovered Segments")
                has_uncovered = True
            status = instruction_segment["covered"]
            report.append(
                f"*   `{instruction_segment['directive']}` ({status})"
            )

    if not has_uncovered:
        report.append("All instruction segments are 100% covered by tests.")
        report.append("")

    report.append("---\n")
    report.append("## Tool Coverage Breakdown\n")
    report.append("### Covered Tools\n")
    if covered_tools:
        for t in sorted(covered_tools):
            report.append(f"*   `{t}`")
    else:
        report.append("*No tools are covered by current evaluations.*")
    report.append("")

    report.append("### Uncovered Tools\n")
    if uncovered_tools:
        for t in sorted(uncovered_tools):
            report.append(f"*   `{t}`")
    else:
        report.append("*All tools are fully covered by evaluations!*")
    report.append("")

    if total_callbacks:
        report.append("---\n")
        report.append("## Callback Coverage Breakdown\n")
        report.append("### Covered Callbacks\n")
        if covered_callbacks:
            for cb in sorted(covered_callbacks):
                report.append(f"*   `{cb}`")
        else:
            report.append("*No callbacks are covered by tests.*")
        report.append("")

        report.append("### Uncovered Callbacks\n")
        uncovered_callbacks = total_callbacks - covered_callbacks
        if uncovered_callbacks:
            for cb in sorted(uncovered_callbacks):
                report.append(f"*   `{cb}`")
        else:
            report.append("*All callbacks are fully covered by tests!*")
        report.append("")

    report.append("---\n")
    report.append("---\n")
    report.append("## Agent Transfer Coverage\n")
    report.append("| From Agent | To Agent | Desired? | Tested? | Eval Names |")
    report.append("| :--- | :--- | :---: | :---: | :--- |")
    for from_a, to_a in declared_transfers:
        desired = "Yes" if (from_a, to_a) in desired_transfers else "No"
        tested = "Yes" if (from_a, to_a) in covered_transfers else "No"
        evals_str = (
            ", ".join(covered_transfers[(from_a, to_a)])
            if (from_a, to_a) in covered_transfers
            else "None"
        )
        report.append(f"| `{from_a}` | `{to_a}` | {desired} | {tested} | {evals_str} |")
    report.append("\n---\n")

    report.append("## Instruction Files Scanned\n")
    for f in instruction_files:
        try:
            rel_f = f.relative_to(agent_dir)
        except ValueError:
            rel_f = f
        report.append(f"*   `{rel_f}`")
    report.append("")

    report.append(
        "### Instruction Segments to Evaluation Files Detailed Mapping\n"
    )
    report.append(
        "| # | Agent | Category | Instruction Quote | Covered? | "
        "Covering Eval(s) |\n"
        "|---|-------|----------|-------------------|----------|"
        "-------------------|"
    )
    for idx, s in enumerate(instruction_segments, start=1):
        report.append(
            f"| {idx} | {s['agent']} | {s['category']} | {s['quote']} | "
            f"{s['covered']} | {s['evals']} |"
        )
    report.append("")

    report.append("---\n")
    report.append("## Scanned Evaluation Files\n")
    if eval_files:
        for ef in sorted(eval_files):
            try:
                rel_ef = ef.relative_to(agent_dir)
            except ValueError:
                rel_ef = ef
            report.append(f"*   `{rel_ef}`")
    else:
        report.append("*No evaluation files scanned.*")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.suffix.lower() == ".json":
        import json

        # Serialize path objects to strings for JSON
        def _path_to_str(p):
            try:
                return str(p.relative_to(agent_dir))
            except ValueError:
                return str(p)

        phantom_tools_str_keys = {
            _path_to_str(k): list(v) for k, v in phantom_tools_by_file.items()
        }

        transfers_list = []
        for from_a, to_a in declared_transfers:
            desired = (from_a, to_a) in desired_transfers
            tested = (from_a, to_a) in covered_transfers
            evals = covered_transfers.get((from_a, to_a), [])
            transfers_list.append({
                "from_agent": from_a,
                "to_agent": to_a,
                "is_desired": desired,
                "is_tested": tested,
                "covering_evals": evals
            })

        json_data = {
            "metrics": {
                "tool_coverage_percent": tool_coverage_pct,
                "instruction_segment_coverage_percent": overall_segment_pct,
                "transfer_coverage_percent": transfer_coverage_pct,
                "callback_coverage_percent": callback_coverage_pct,
                "total_tools": len(total_tools),
                "covered_tools": len(covered_tools),
                "total_segments": total_segments,
                "covered_segments": total_covered,
                "total_transfers": total_transfers,
                "covered_transfers": total_transfers_covered,
                "total_callbacks": total_cbs,
                "covered_callbacks": covered_cbs,
                "category_counts": category_counts,
                "category_covered_counts": category_covered_counts,
            },
            "errors": errors if errors else [],
            "phantom_tools_by_file": phantom_tools_str_keys,
            "tools": {
                "covered": sorted(list(covered_tools)),
                "uncovered": sorted(list(uncovered_tools))
            },
            "callbacks": {
                "covered": sorted(list(covered_callbacks)),
                "uncovered": sorted(list(total_callbacks - covered_callbacks))
            },
            "agent_transfers": transfers_list,
            "scanned_files": {
                "instructions": [_path_to_str(f) for f in instruction_files],
                "evaluations": [_path_to_str(f) for f in eval_files]
            },
            "instruction_segments": instruction_segments,
            "covered_instruction_segments": covered_instruction_segments
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
    else:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report))

    print(f"Successfully generated coverage report at: {output_file}")


async def main():
    parser = argparse.ArgumentParser(description="Calculate eval coverage.")
    parser.add_argument(
        "--agent-dir",
        required=True,
        help="Directory path to GECX agent project.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="File path to save markdown coverage report or JSON.",
    )
    parser.add_argument(
        "--project-id",
        help="Google Cloud Project ID for Gemini embeddings and LLM judge.",
    )
    parser.add_argument(
        "--location",
        default="global",
        help="Google Cloud location for Gemini services (default: global).",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name to use for analysis (default: gemini-2.5-flash).",
    )
    args = parser.parse_args()

    agent_dir = Path(args.agent_dir)
    output_file = Path(args.output_file)

    if args.project_id:
        os.environ["GOOGLE_CLOUD_PROJECT"] = args.project_id
    if args.location:
        os.environ["GOOGLE_CLOUD_LOCATION"] = args.location

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get(
        "GCP_PROJECT"
    )
    location = os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"

    print(f"Initializing Gemini Generate for (Project: {project_id})...")
    gemini_client = GeminiGenerate(
        project_id=project_id,
        location=location,
        model_name=args.model,
    )

    execution_errors = []

    # 1. Unified Ingestion Pass (Reads files once, chunks everything)
    print(f"Ingesting and parsing agent workspace at: {agent_dir}...")
    agent_data = ingest_agent_project(agent_dir)

    print("Determining desired agent transfers with LLM...")
    agent_data.desired_transfers = await determine_desired_transfers_with_llm(
        agent_data.agent_directories,
        agent_data.declared_transfers,
        gemini_client,
        errors=execution_errors,
    )

    # Automatically mark every parent to child transfer as desired
    agent_data.desired_transfers.update(agent_data.parent_child_transfers)

    # 2. Run classification pass on instruction segments
    agent_data.instruction_segments = await analyze_instruction_categories(
        agent_data.instruction_segments, gemini_client, errors=execution_errors
    )

    # Filter out untestable segments
    testable_segments = [s for s in agent_data.instruction_segments if s.get("is_testable", True)]

    # 3. Run instruction coverage analysis pass
    instruction_segments, covered_instruction_segments = (
        await extract_instruction_coverage(
            testable_segments,
            agent_data.eval_chunks,
            agent_data.called_tools,
            gemini_client,
            errors=execution_errors,
        )
    )

    # 4. Warnings for phantoms
    if agent_data.phantom_tools_by_file:
        print(
            "\n[WARNING] Detected tools that are referenced in evaluations "
            "but do not exist in the tools directory:"
        )
        for ef, phantoms in sorted(agent_data.phantom_tools_by_file.items()):
            try:
                rel_path = ef.relative_to(agent_dir)
            except ValueError:
                rel_path = ef
            print(f"  - {rel_path}: {', '.join(sorted(phantoms))}")
        print(
            "Please verify if these tools were renamed, deleted, or "
            "misspelled.\n"
        )

    # 5. Generate clean report
    generate_report(
        output_file=output_file,
        total_tools=agent_data.all_tools,
        covered_tools=agent_data.covered_tools,
        phantom_tools_by_file=agent_data.phantom_tools_by_file,
        eval_files=agent_data.eval_files,
        declared_transfers=agent_data.declared_transfers,
        covered_transfers=agent_data.covered_transfers,
        instruction_segments=instruction_segments,
        covered_instruction_segments=covered_instruction_segments,
        instruction_files=agent_data.instruction_files,
        agent_dir=agent_dir,
        total_callbacks=agent_data.all_callbacks,
        covered_callbacks=agent_data.covered_callbacks,
        desired_transfers=agent_data.desired_transfers,
        errors=execution_errors,
    )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
