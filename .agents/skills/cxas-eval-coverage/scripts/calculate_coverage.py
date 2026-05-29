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

"""Calculates evaluation coverage metrics for a GECX conversational agent."""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml
from pydantic import BaseModel, Field

from cxas_scrapi.utils.gemini import GeminiGenerate


# Functions to evaluate tool coverage
def find_tools(tools_dir: Path) -> Set[str]:
    """Finds all declared tool names in the tools directory."""
    tools = set()
    if not tools_dir.exists():
        return tools

    # Look for json or yaml tool definitions
    for p in tools_dir.glob("**/*"):
        if p.suffix in (".json", ".yaml", ".yml"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    content = (
                        yaml.safe_load(f)
                        if p.suffix in (".yaml", ".yml")
                        else json.load(f)
                    )

                # Parse GECX tool structure if it has tool definitions
                # Typically the displayName or filename (excluding ext)
                # represents the tool name
                tool_name = p.stem
                if isinstance(content, dict):
                    if "displayName" in content:
                        tool_name = content["displayName"]
                    elif "name" in content:
                        name_val = content["name"]
                        # If it is not a UUID, we can use it
                        if not re.match(r"^[0-9a-fA-F-]{36}$", str(name_val)):
                            tool_name = name_val
                tools.add(tool_name)
            except Exception:  # pylint: disable=broad-except
                # Fallback to filename if parsing fails
                tools.add(p.stem)
    return tools


def parse_golden_evals(eval_file: Path) -> Set[str]:
    """Parses tool calls from a SCRAPI Golden Eval YAML file."""
    called_tools = set()
    try:
        with open(eval_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data or "conversations" not in data:
            return called_tools

        for conv in data["conversations"]:
            for turn in conv.get("turns", []):
                for tool_call in turn.get("tool_calls", []):
                    if isinstance(tool_call, dict) and "action" in tool_call:
                        called_tools.add(tool_call["action"])
    except Exception as e:
        print(f"Warning: Failed to parse golden eval {eval_file}: {e}")
    return called_tools


def parse_native_json_evals(eval_file: Path) -> Set[str]:
    """Parses tool calls from GECX native JSON evaluations."""
    called_tools = set()
    try:
        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return called_tools

        golden = data.get("golden")
        if not isinstance(golden, dict):
            return called_tools

        turns = golden.get("turns", [])
        for turn in turns:
            steps = turn.get("steps", [])
            for step in steps:
                expectation = step.get("expectation")
                if isinstance(expectation, dict):
                    tool_call = expectation.get("toolCall")
                    if isinstance(tool_call, dict) and "tool" in tool_call:
                        called_tools.add(tool_call["tool"])
    except Exception as e:
        print(f"Warning: Failed to parse native json eval {eval_file}: {e}")
    return called_tools


def parse_simulation_evals(eval_file: Path, all_tools: Set[str]) -> Set[str]:
    """Parses tool calls and referenced tools from Simulation Eval YAML file."""
    referenced_tools = set()
    try:
        with open(eval_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data or "evals" not in data:
            return referenced_tools

        for eval_item in data["evals"]:
            # Scan expectations and success criteria for tool names
            expectations = eval_item.get("expectations", [])
            steps = eval_item.get("steps", [])

            text_to_scan = []
            text_to_scan.extend(expectations)
            for step in steps:
                if "success_criteria" in step:
                    text_to_scan.append(step["success_criteria"])
                if "goal" in step:
                    text_to_scan.append(step["goal"])

            for text in text_to_scan:
                if not isinstance(text, str):
                    continue
                # Look for mentions of known tools in the expectation text
                for tool in all_tools:
                    # Use word boundaries to avoid matching partial substrings
                    if re.search(
                        rf"\b{re.escape(tool)}\b", text, re.IGNORECASE
                    ):
                        referenced_tools.add(tool)
    except Exception as e:
        print(f"Warning: Failed to parse simulation eval {eval_file}: {e}")
    return referenced_tools


# Function to evaluate agent transfer coverage
def extract_agent_transfers(
    agent_dir: Path, eval_files: List[Path]
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], List[str]]]:
    """Extracts declared and covered agent transfers from agents and
    evaluations."""
    declared_transfers = []
    covered_transfers = {}
    agent_files = list((agent_dir / "agents").glob("**/*.json"))

    agents = {}
    root_agents = set()
    for af in agent_files:
        try:
            with open(af, "r", encoding="utf-8") as f:
                data = json.load(f)
            display_name = data.get("displayName")
            agents[display_name] = data
            root_agents.add(display_name)
        except Exception:
            pass

    for display_name, data in agents.items():
        children = data.get("childAgents", [])
        for child in children:
            root_agents.discard(child)
            declared_transfers.append((display_name, child))
            for c2 in children:
                if child != c2:
                    declared_transfers.append((child, c2))

    for ef in eval_files:
        if ef.suffix == ".json":
            try:
                with open(ef, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                eval_name = (
                    eval_data.get("displayName")
                    or eval_data.get("name")
                    or ef.name
                )
                target_agents = []

                def find_target_agent(obj, ta_list):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if k == "targetAgent":
                                ta_list.append(v)
                            else:
                                find_target_agent(v, ta_list)
                    elif isinstance(obj, list):
                        for item in obj:
                            find_target_agent(item, ta_list)

                find_target_agent(eval_data, target_agents)

                current_agent = next(iter(root_agents)) if root_agents else None
                for target in target_agents:
                    if current_agent and target:
                        edge = (current_agent, target)
                        if edge not in covered_transfers:
                            covered_transfers[edge] = []
                        if eval_name not in covered_transfers[edge]:
                            covered_transfers[edge].append(eval_name)
                        current_agent = target
            except Exception:
                pass

    return declared_transfers, covered_transfers


def analyze_instructions(
    agent_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Path]]:
    """Discovers and parses all instruction.txt and instruction.* files
    into instruction_segments."""
    instruction_segments = []
    instruction_files = []

    def add_instruction_segment(quote_lines, cat_name, a_name):
        q_text = " ".join(quote_lines).strip()
        if len(q_text) > 10:
            q_text = re.sub(r"^\d+[\.\)]\s*", "", q_text)
            q_text = re.sub(r"^[\-\*]\s*", "", q_text)
            q_text = q_text.strip()
            directive_title = " ".join(q_text.split()[:5])
            if len(directive_title) < len(q_text):
                directive_title += "..."
            instruction_segments.append(
                {
                    "agent": a_name,
                    "category": cat_name,
                    "directive": directive_title,
                    "quote": f'"{q_text[:60]}..."'
                    if len(q_text) > 60
                    else f'"{q_text}"',
                    "full_text": q_text,
                }
            )

    def parse_file(filepath: Path, agent_name: str):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            sections = re.findall(
                r"<([a-zA-Z0-9_-]+)>(.*?)</\1>", content, re.DOTALL
            )

            for tag, text in sections:
                lines = text.split("\n")
                category = "Rules"
                tag_lower = tag.lower()
                if tag_lower in ("role", "persona", "voice"):
                    category = "Persona"
                elif tag_lower in (
                    "guidelines",
                    "guideline",
                    "out_of_scope",
                    "condition",
                    "trigger",
                ):
                    category = (
                        "Cond. Behavior"
                        if tag_lower in ("condition", "trigger")
                        else "Guardrails"
                    )
                elif tag_lower in ("taskflow", "subtask", "action", "step"):
                    category = "Conv. Flow"

                current_quote = []
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if (
                        re.match(r"^\d+[\.\)]\s*", stripped)
                        or stripped.startswith("-")
                        or stripped.startswith("*")
                    ):
                        if current_quote:
                            add_instruction_segment(
                                current_quote, category, agent_name
                            )
                            current_quote = [stripped]
                        else:
                            current_quote = [stripped]
                    else:
                        current_quote.append(stripped)
                if current_quote:
                    add_instruction_segment(current_quote, category, agent_name)

            if not sections:
                lines = content.split("\n")
                current_quote = []
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if (
                        re.match(r"^\d+[\.\)]\s*", stripped)
                        or stripped.startswith("-")
                        or stripped.startswith("*")
                    ):
                        if current_quote:
                            add_instruction_segment(
                                current_quote, "Rules", agent_name
                            )
                            current_quote = [stripped]
                        else:
                            current_quote = [stripped]
                    else:
                        current_quote.append(stripped)
                if current_quote:
                    add_instruction_segment(current_quote, "Rules", agent_name)
        except Exception as e:
            print(f"Warning: Failed to parse instructions {filepath}: {e}")

    # Look for sub-agent instruction files recursively
    agents_dir = agent_dir / "agents"
    if agents_dir.exists() and agents_dir.is_dir():
        for p in agents_dir.glob("**/instruction.*"):
            if p.is_file():
                instruction_files.append(p)
                parse_file(p, p.parent.name)

    p = agent_dir / "global_instruction.txt"
    if p.is_file():
        instruction_files.append(p)
        parse_file(p, "Global")

    return instruction_segments, instruction_files


def extract_instruction_coverage(
    instruction_segments: List[Dict[str, Any]],
    eval_files: List[Path],
    called_tools: Set[str],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Uses Vector Embeddings and LLM-as-a-judge to determine instruction
    instruction_segment coverage."""

    eval_chunks = []
    for ef in eval_files:
        try:
            eval_name = ef.stem
            if ef.suffix in (".yaml", ".yml"):
                with open(ef, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if not data:
                    continue

                # SCRAPI Golden Evals
                if "conversations" in data:
                    for conv in data.get("conversations", []):
                        c_name = conv.get("conversation", "Unnamed")
                        tags = conv.get("tags", [])

                        turns_text = []
                        for turn in conv.get("turns", []):
                            user = turn.get("user", "")
                            agent = turn.get("agent", "")
                            turn_str = f"User: {user}\nAgent: {agent}"
                            if "tool_calls" in turn:
                                turn_str += (
                                    f"\nTool Calls: "
                                    f"{json.dumps(turn['tool_calls'])}"
                                )
                            turns_text.append(turn_str)

                        if turns_text:
                            eval_chunks.append(
                                {
                                    "text": (
                                        f"Conversation: {c_name}\n"
                                        f"Tags: {', '.join(tags)}\n"
                                        + "\n".join(turns_text)
                                    ),
                                    "eval_name": c_name or eval_name,
                                    "file_name": ef.name,
                                }
                            )

                        expectations = conv.get("expectations", [])
                        if expectations:
                            eval_chunks.append(
                                {
                                    "text": (
                                        f"Conversation: {c_name}\n"
                                        "Expectations:\n"
                                        + "\n".join(
                                            f"- {exp}" for exp in expectations
                                        )
                                    ),
                                    "eval_name": c_name or eval_name,
                                    "file_name": ef.name,
                                }
                            )

                # SCRAPI Simulation Evals
                elif "evals" in data:
                    for eval_item in data.get("evals", []):
                        e_name = eval_item.get("name", "Unnamed")
                        tags = eval_item.get("tags", [])

                        steps_text = []
                        for step in eval_item.get("steps", []):
                            goal = step.get("goal", "")
                            success = step.get("success_criteria", "")
                            guide = step.get("response_guide", "")
                            steps_text.append(
                                f"Goal: {goal}\nSuccess Criteria: "
                                f"{success}\nResponse Guide: {guide}"
                            )

                        if steps_text:
                            eval_chunks.append(
                                {
                                    "text": (
                                        f"Simulation Eval: {e_name}\n"
                                        f"Tags: {', '.join(tags)}\n"
                                        + "\n".join(steps_text)
                                    ),
                                    "eval_name": e_name or eval_name,
                                    "file_name": ef.name,
                                }
                            )

                        expectations = eval_item.get("expectations", [])
                        if expectations:
                            eval_chunks.append(
                                {
                                    "text": (
                                        f"Simulation Eval: {e_name}\n"
                                        "Expectations:\n"
                                        + "\n".join(
                                            f"- {exp}" for exp in expectations
                                        )
                                    ),
                                    "eval_name": e_name or eval_name,
                                    "file_name": ef.name,
                                }
                            )

            elif ef.suffix == ".json":
                with open(ef, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    eval_name = (
                        data.get("displayName") or data.get("name") or ef.stem
                    )
                    golden = data.get("golden", {})
                    for conv_idx, turn in enumerate(golden.get("turns", [])):
                        steps = turn.get("steps", [])
                        turn_text = []
                        for step in steps:
                            if "userInput" in step:
                                turn_text.append(
                                    f"User: {step['userInput'].get('text', '')}"
                                )
                            if "expectation" in step:
                                exp = step["expectation"]
                                if "note" in exp:
                                    turn_text.append(
                                        f"Expectation Note: {exp['note']}"
                                    )
                                if "agentTransfer" in exp:
                                    target_ag = exp["agentTransfer"].get(
                                        "targetAgent", ""
                                    )
                                    turn_text.append(
                                        f"Expects Transfer to: {target_ag}"
                                    )
                                if "toolCall" in exp:
                                    turn_text.append(
                                        "Expects Tool Call: "
                                        f"{exp['toolCall'].get('tool', '')}"
                                    )
                                if "updatedVariables" in exp:
                                    turn_text.append(
                                        "Expects Updated Variables: "
                                        f"{json.dumps(exp['updatedVariables'])}"
                                    )

                        if turn_text:
                            eval_chunks.append(
                                {
                                    "text": (
                                        f"Native Eval: {eval_name} "
                                        f"(Turn {conv_idx})\n"
                                        + "\n".join(turn_text)
                                    ),
                                    "eval_name": eval_name,
                                    "file_name": ef.name,
                                }
                            )
        except Exception as e:
            print(f"Warning: Failed to chunk evaluation file {ef}: {e}")

    # Helper functions for cosine similarity
    def dot_product(v1, v2):
        """Calculates the dot product of two vectors."""
        return sum(a * b for a, b in zip(v1, v2, strict=True))

    def magnitude(v):
        """Calculates the Euclidean magnitude of a vector."""
        return math.sqrt(sum(a * a for a in v))

    def cosine_similarity(v1, v2):
        """Calculates the cosine similarity between two vectors."""
        mag1 = magnitude(v1)
        mag2 = magnitude(v2)
        if not mag1 or not mag2:
            return 0.0
        return dot_product(v1, v2) / (mag1 * mag2)

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get(
        "GCP_PROJECT"
    )
    location = os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"

    print(f"Initializing Gemini Generate for (Project: {project_id})...")
    gemini_client = GeminiGenerate(
        project_id=project_id,
        location=location,
        model_name="gemini-2.5-flash",
    )

    instruction_segments_texts = [
        instruction_segment["full_text"]
        for instruction_segment in instruction_segments
    ]
    chunk_texts = [chunk["text"] for chunk in eval_chunks]

    instruction_segment_embeddings = []
    chunk_embeddings = []

    if instruction_segments_texts:
        print(
            f"""Generating embeddings for {len(instruction_segments_texts)}
            instruction segment(s)..."""
        )
        instruction_segment_embeddings = gemini_client.generate_embeddings(
            contents=instruction_segments_texts
        )

    if chunk_texts:
        print(f"Generating embeddings for {len(chunk_texts)} eval chunk(s)...")
        chunk_embeddings = gemini_client.generate_embeddings(
            contents=chunk_texts
        )

    class InstructionSegmentCoverageResult(BaseModel):
        """Schema for the LLM evaluation of instruction segment coverage."""

        is_covered: bool = Field(
            description="""true if at least one evaluation chunk explicitly
            tests the instruction, false otherwise."""
        )
        covering_chunk_index: int = Field(
            description="""The 0-based index of the candidate chunk that tests
                the instruction. Set to -1 if none."""
        )
        reasoning: str = Field(
            description="""A brief reasoning string explaining the decision."""
        )

    covered_instruction_segments = []
    uncovered_instruction_segments = []

    print("Evaluating instruction coverage using Cosine Similarity and LLM...")
    for i, instruction_segment in enumerate(instruction_segments):
        covered = False
        covering_evals = set()

        # 1. Fallback / Immediate tool-call matching
        text_to_check = instruction_segment["full_text"].lower()
        match_tool = re.search(r"\{@TOOL[:\s]+([^}]+)\}", text_to_check)
        if match_tool:
            tool_name = match_tool.group(1).strip()
            if tool_name in called_tools:
                covered = True
                for ef in eval_files:
                    try:
                        with open(ef, "r", encoding="utf-8") as f:
                            if tool_name in f.read():
                                eval_name = ef.stem
                                covering_evals.add(eval_name)
                    except Exception:
                        pass

        # 2. Vector Embeddings + LLM Judge
        if (
            not covered
            and i < len(instruction_segment_embeddings)
            and instruction_segment_embeddings[i]
            and chunk_embeddings
        ):
            i_embedding = instruction_segment_embeddings[i]
            similarities = []
            for j, c_embedding in enumerate(chunk_embeddings):
                if c_embedding:
                    sim = cosine_similarity(i_embedding, c_embedding)
                    similarities.append((sim, j))
                else:
                    similarities.append((0.0, j))

            # Get top 4 most relevant chunks
            similarities.sort(reverse=True, key=lambda x: x[0])
            top_candidates = similarities[:4]

            candidate_chunks = [
                eval_chunks[idx] for sim, idx in top_candidates if sim > 0.0
            ]

            if candidate_chunks:
                chunks_formatted_text = ""
                for idx, c in enumerate(candidate_chunks):
                    chunks_formatted_text += (
                        f"\n--- CANDIDATE CHUNK {idx} ---\n{c['text']}\n"
                    )

                prompt = f"""
                You are an expert LLM as a Judge determining evaluation coverage
                for an AI Agent.

                Agent Instruction to Test:
                <INSTRUCTION>
                {instruction_segment["full_text"]}
                </INSTRUCTION>

                Candidate Evaluation Chunks:
                {chunks_formatted_text}

                Analyze the Candidate Evaluation Chunks carefully.
                Determine if ANY of these evaluation chunks explicitly test that
                the Agent follows the provided Agent Instruction.
                Answer true in `is_covered` if at least one evaluation chunk
                explicitly tests the instruction, and identify the FIRST
                covering chunk's index (0-based) in `covering_chunk_index`.
                General instructions like "be nice" should be considered as
                covered.
                """
                try:
                    llm_response = gemini_client.generate(
                        prompt=prompt,
                        response_mime_type="application/json",
                        response_schema=InstructionSegmentCoverageResult,
                        temperature=0.0,
                    )

                    if llm_response:
                        is_cov = getattr(llm_response, "is_covered", False)
                        c_idx = getattr(
                            llm_response, "covering_chunk_index", -1
                        )

                        if isinstance(llm_response, dict):
                            is_cov = llm_response.get("is_covered", False)
                            c_idx = llm_response.get("covering_chunk_index", -1)

                        if is_cov and 0 <= c_idx < len(candidate_chunks):
                            covered = True
                            covering_chunk = candidate_chunks[c_idx]
                            covering_evals.add(covering_chunk["eval_name"])
                except Exception as e:
                    print(f"LLM call failed for instruction segment {i}: {e}")

        instruction_segment["covered"] = "Yes" if covered else "No"
        instruction_segment["evals"] = (
            ", ".join(sorted(covering_evals)) if covering_evals else "None"
        )
        if covered:
            covered_instruction_segments.append(instruction_segment)
        else:
            uncovered_instruction_segments.append(instruction_segment)

    return instruction_segments, covered_instruction_segments


def generate_report(
    output_file: Path,
    total_tools: Set[str],
    covered_tools: Set[str],
    phantom_tools_by_file: dict[Path, Set[str]],
    eval_files: List[Path],
    declared_transfers: List[Tuple[str, str]],
    covered_transfers: Dict[Tuple[str, str], List[str]],
    instruction_segments: List[Dict[str, Any]],
    covered_instruction_segments: List[Dict[str, Any]],
    instruction_files: List[Path],
    agent_dir: Path,
) -> None:
    """Generates a comprehensive Markdown coverage report."""
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

    segment_counts = {"Static": 0, "Stateful": 0, "Conditional": 0, "Tool": 0}
    segment_covered = {"Static": 0, "Stateful": 0, "Conditional": 0, "Tool": 0}

    for instruction_segment in instruction_segments:
        cat = "Static"
        if instruction_segment["category"] in ("Cond. Behavior", "Guardrails"):
            cat = "Conditional"
        elif instruction_segment["category"] in ("Conv. Flow",):
            cat = "Stateful"
        elif instruction_segment["category"] == "Tool":
            cat = "Tool"

        segment_counts[cat] += 1
        if instruction_segment["covered"] == "Yes":
            segment_covered[cat] += 1

    report = []
    report.append("# Evaluation Coverage Report\n")

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
    report.append("\n")

    report.append("## Instruction segment Category Breakdown\n")
    report.append("| Category | Description | Total | Covered | Coverage % |")
    report.append("| :--- | :--- | :---: | :---: | :---: |")

    static_pct = (
        (segment_covered["Static"] / segment_counts["Static"] * 100.0)
        if segment_counts["Static"]
        else 0.0
    )
    report.append(
        "| **Static segments** | "
        "Global Instructions, Persona, Voice, Guidelines | "
        f"{segment_counts['Static']} | {segment_covered['Static']} | "
        f"{static_pct:.1f}% |"
    )

    stateful_pct = (
        (segment_covered["Stateful"] / segment_counts["Stateful"] * 100.0)
        if segment_counts["Stateful"]
        else 0.0
    )
    report.append(
        f"| **Stateful segments** | "
        "Context-dependent states (Slots)| "
        f"{segment_counts['Stateful']} | "
        f"{segment_covered['Stateful']} | "
        f"{stateful_pct:.1f}% |"
    )

    cond_pct = (
        (segment_covered["Conditional"] / segment_counts["Conditional"] * 100.0)
        if segment_counts["Conditional"]
        else 0.0
    )
    report.append(
        "| **Conditional segments** | "
        "Branching logic Conditions, Routing | "
        f"{segment_counts['Conditional']} | "
        f"{segment_covered['Conditional']} | "
        f"{cond_pct:.1f}% |"
    )

    report.append("\n---\n")

    report.append("## Uncovered segments\n")
    has_uncovered = False
    for instruction_segment in instruction_segments:
        if instruction_segment["covered"] == "No":
            if not has_uncovered:
                report.append("### Uncovered Segments")
                has_uncovered = True
            report.append(f"*   `{instruction_segment['directive']}`")

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

    report.append("---\n")
    report.append("---\n")
    report.append("## Agent Transfer Coverage\n")
    report.append("| From Agent | To Agent | Tested? | Eval Names |")
    report.append("| :--- | :--- | :---: | :--- |")
    for from_a, to_a in declared_transfers:
        tested = "Yes" if (from_a, to_a) in covered_transfers else "No"
        evals_str = (
            ", ".join(covered_transfers[(from_a, to_a)])
            if (from_a, to_a) in covered_transfers
            else "None"
        )
        report.append(f"| `{from_a}` | `{to_a}` | {tested} | {evals_str} |")
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
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Successfully generated coverage report at: {output_file}")


def run_coverage_analysis(
    agent_dir: Path,
) -> Tuple[
    Set[str],
    Set[str],
    Dict[Path, Set[str]],
    List[Path],
    Set[str],
]:
    tools_dir = agent_dir / "tools"
    eval_dir = agent_dir / "evaluations"
    eval_dataset_dir = agent_dir / "evaluationDatasets"

    all_tools = find_tools(tools_dir)

    eval_files = []
    for d in (eval_dir, eval_dataset_dir):
        if d.exists():
            for p in d.glob("**/*"):
                if p.is_file() and p.suffix in (".json", ".yaml", ".yml"):
                    eval_files.append(p)

    covered_tools = set()
    phantom_tools_by_file = {}
    eval_texts = set()
    called_tools = set()

    for ef in eval_files:
        file_tools = set()
        if ef.suffix == ".json":
            native_tools = parse_native_json_evals(ef)
            file_tools.update(native_tools)
            try:
                with open(ef, "r", encoding="utf-8") as f:
                    eval_texts.add(f.read())
            except Exception:
                pass
        elif ef.suffix in (".yaml", ".yml"):
            gold_tools = parse_golden_evals(ef)
            file_tools.update(gold_tools)

            sim_tools = parse_simulation_evals(ef, all_tools)
            file_tools.update(sim_tools)

            try:
                with open(ef, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f)
                if yaml_data:
                    eval_texts.add(json.dumps(yaml_data))
            except Exception:
                pass

        phantoms = file_tools - all_tools - {"end_session"}
        if phantoms:
            phantom_tools_by_file[ef] = phantoms

        covered_tools.update(file_tools & all_tools)
        called_tools.update(file_tools)

    return (
        all_tools,
        covered_tools,
        phantom_tools_by_file,
        eval_files,
        called_tools,
    )


def main():
    parser = argparse.ArgumentParser(description="Calculate eval coverage.")
    parser.add_argument(
        "--agent-dir",
        required=True,
        help="Directory path to GECX agent project.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="File path to save markdown coverage report.",
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
    args = parser.parse_args()

    agent_dir = Path(args.agent_dir)
    output_file = Path(args.output_file)

    if args.project_id:
        os.environ["GOOGLE_CLOUD_PROJECT"] = args.project_id
    if args.location:
        os.environ["GOOGLE_CLOUD_LOCATION"] = args.location

    (
        all_tools,
        covered_tools,
        phantom_tools_by_file,
        eval_files,
        called_tools,
    ) = run_coverage_analysis(agent_dir)

    declared_transfers, covered_transfers = extract_agent_transfers(
        agent_dir, eval_files
    )

    instruction_segments, instruction_files = analyze_instructions(agent_dir)

    instruction_segments, covered_instruction_segments = (
        extract_instruction_coverage(
            instruction_segments, eval_files, called_tools
        )
    )

    if phantom_tools_by_file:
        print(
            "\n[WARNING] Detected tools that are referenced in evaluations "
            "but do not exist in the tools directory:"
        )
        for ef, phantoms in sorted(phantom_tools_by_file.items()):
            try:
                rel_path = ef.relative_to(agent_dir)
            except ValueError:
                rel_path = ef
            print(f"  - {rel_path}: {', '.join(sorted(phantoms))}")
        print(
            "Please verify if these tools were renamed, deleted, or "
            "misspelled.\n"
        )

    generate_report(
        output_file,
        all_tools,
        covered_tools,
        phantom_tools_by_file,
        eval_files,
        declared_transfers,
        covered_transfers,
        instruction_segments,
        covered_instruction_segments,
        instruction_files,
        agent_dir,
    )


if __name__ == "__main__":
    main()
