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
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml


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


class InstructionSegment:
    def __init__(
        self,
        tag_name: str,
        attributes: Dict[str, str],
        content: str,
        start: int,
        end: int,
    ):
        self.tag_name = tag_name
        self.attributes = attributes
        self.content = content
        self.start = start
        self.end = end
        self.children: List["InstructionSegment"] = []
        self.id = ""
        self.type = ""
        self.category = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "category": self.category,
            "tag": self.tag_name,
            "attributes": self.attributes,
            "content": self.content.strip() if not self.children else "",
            "children": [c.to_dict() for c in self.children],
        }


def extract_keywords(text: str) -> Set[str]:
    """Extracts unique lowercase alphanumeric keywords of length >= 3, ignoring common stopwords."""
    stopwords = {
        "the",
        "and",
        "you",
        "are",
        "our",
        "for",
        "with",
        "can",
        "this",
        "that",
        "your",
        "have",
        "has",
        "not",
        "but",
        "will",
        "shall",
        "should",
        "would",
        "could",
        "please",
        "any",
        "some",
        "all",
        "say",
        "tell",
        "get",
        "take",
        "make",
        "call",
    }
    words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
    return {w for w in words if w not in stopwords}


def parse_instruction_to_tree(content: str) -> List[InstructionSegment]:
    """Recursively parses GECX instruction.txt into a tree of InstructionSegments."""

    def parse_recursive(
        text: str, start_offset: int = 0
    ) -> List[InstructionSegment]:
        nodes = []
        pos = 0
        while pos < len(text):
            # Look for next start tag
            start_match = re.search(
                r"<([a-zA-Z0-9_-]+)(?:\s+([^>]+))?>", text[pos:]
            )
            if not start_match:
                break

            tag_name = start_match.group(1)
            attr_str = start_match.group(2) or ""
            start_tag_index = pos + start_match.start()
            after_start_tag = pos + start_match.end()

            attrs = {}
            if attr_str:
                attr_matches = re.findall(
                    r'([a-zA-Z0-9_-]+)="([^"]*)"', attr_str
                )
                attrs = {k: v for k, v in attr_matches}

            # Find matching end tag, respecting nested tags of same type
            end_tag_pattern = re.compile(rf"</{tag_name}>")
            start_tag_pattern = re.compile(rf"<{tag_name}(?:\s+[^>]+)?>")

            scan_pos = after_start_tag
            nest_level = 1
            end_tag_index = -1
            after_end_tag = -1

            while nest_level > 0:
                next_end = end_tag_pattern.search(text, scan_pos)
                if not next_end:
                    end_tag_index = len(text)
                    after_end_tag = len(text)
                    break

                next_start = start_tag_pattern.search(text, scan_pos)
                if next_start and next_start.start() < next_end.start():
                    nest_level += 1
                    scan_pos = next_start.end()
                else:
                    nest_level -= 1
                    if nest_level == 0:
                        end_tag_index = next_end.start()
                        after_end_tag = next_end.end()
                    else:
                        scan_pos = next_end.end()

            tag_content = text[after_start_tag:end_tag_index]
            node = InstructionSegment(
                tag_name=tag_name,
                attributes=attrs,
                content=tag_content,
                start=start_offset + start_tag_index,
                end=start_offset + after_end_tag,
            )

            node.children = parse_recursive(
                tag_content, start_offset + after_start_tag
            )

            # Also discover dynamic inline tool and handoff segments inside content
            # if there are no sub-tags already parsed.
            if not node.children and tag_name.lower() in ("action", "step"):
                # Find all {@TOOL: ToolName}
                tool_refs = re.findall(
                    r"\{@TOOL[:\s]+([^}]+)\}", tag_content, re.IGNORECASE
                )
                for tool_name in tool_refs:
                    t_node = InstructionSegment(
                        tag_name="tool_ref",
                        attributes={"name": tool_name.strip()},
                        content="",
                        start=node.start,
                        end=node.end,
                    )
                    node.children.append(t_node)

                # Find all {@AGENT: AgentName}
                agent_refs = re.findall(
                    r"\{@AGENT[:\s]+([^}]+)\}", tag_content, re.IGNORECASE
                )
                for agent_name in agent_refs:
                    a_node = InstructionSegment(
                        tag_name="handoff",
                        attributes={"name": agent_name.strip()},
                        content="",
                        start=node.start,
                        end=node.end,
                    )
                    node.children.append(a_node)

            nodes.append(node)
            pos = after_end_tag

        return nodes

    root_nodes = parse_recursive(content)

    # Assign IDs, Types, and Categories hierarchically
    def assign_metadata(node: InstructionSegment, parent_id: str = ""):
        tag_lower = node.tag_name.lower()

        # Assign Type
        type_map = {
            "role": "Persona",
            "persona": "Persona",
            "voice": "Voice",
            "guideline": "Rule",
            "global_rules": "RuleGroup",
            "rule": "Rule",
            "taskflow": "Taskflow",
            "subtask": "Sequence",
            "step": "SequenceStep",
            "trigger": "Trigger",
            "action": "Action",
            "condition": "Condition",
            "handoff": "Handoff",
            "slot": "Slot",
            "tool_ref": "Tool",
        }
        node.type = type_map.get(tag_lower, "segment")

        # Assign Category
        if tag_lower in (
            "role",
            "persona",
            "voice",
            "guideline",
            "global_rules",
            "rule",
            "guidelines",
        ):
            node.category = "Static"
        elif tag_lower in ("subtask", "step", "slot", "trigger", "action"):
            node.category = "Stateful"
        elif tag_lower in ("condition", "handoff", "out_of_scope"):
            node.category = "Conditional"
        elif tag_lower in ("tool_ref",):
            node.category = "Tool"
        else:
            node.category = "Stateful"

        name_attr = node.attributes.get("name", "")

        if tag_lower in (
            "role",
            "persona",
            "voice",
            "taskflow",
            "global_rules",
            "guidelines",
        ):
            node.id = tag_lower
        elif tag_lower == "subtask" and name_attr:
            node.id = f"taskflow.subtask.{name_attr}"
        else:
            suffix = f".{name_attr}" if name_attr else ""
            node.id = (
                f"{parent_id}.{tag_lower}{suffix}"
                if parent_id
                else f"{tag_lower}{suffix}"
            )

        for child in node.children:
            assign_metadata(child, node.id)

    for r in root_nodes:
        assign_metadata(r)

    return root_nodes


def flatten_instruction_tree(
    nodes: List[InstructionSegment],
) -> Dict[str, InstructionSegment]:
    """Flattens tree nodes into a dictionary mapping ID to InstructionSegment."""
    flat = {}

    def traverse(node: InstructionSegment):
        flat[node.id] = node
        for child in node.children:
            traverse(child)

    for n in nodes:
        traverse(n)
    return flat


def analyze_instructions(
    agent_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, InstructionSegment]]:
    """Discovers and parses all instruction.txt files into a consolidated segment tree."""
    instruction_files = []

    p = agent_dir / "instruction.txt"
    if p.is_file():
        instruction_files.append(p)

    # Look for sub-agent instruction files recursively
    agents_dir = agent_dir / "agents"
    if agents_dir.exists() and agents_dir.is_dir():
        for p in agents_dir.glob("**/instruction.txt"):
            if p.is_file():
                instruction_files.append(p)

    all_tree_dicts = []
    all_flat_segments = {}

    for inst_file in instruction_files:
        try:
            with open(inst_file, "r", encoding="utf-8") as f:
                content = f.read()

            tree_nodes = parse_instruction_to_tree(content)
            for n in tree_nodes:
                all_tree_dicts.append(n.to_dict())

            flat = flatten_instruction_tree(tree_nodes)
            all_flat_segments.update(flat)
        except Exception as e:
            print(f"Warning: Failed to parse instructions {inst_file}: {e}")

    return all_tree_dicts, all_flat_segments


def extract_dynamic_coverage(
    agent_dir: Path, eval_files: List[Path], called_tools: Set[str]
) -> Tuple[
    List[Tuple[str, str]],
    Dict[Tuple[str, str], List[str]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Path],
]:
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

                def find_target_agent(obj):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if k == "targetAgent":
                                target_agents.append(v)
                            else:
                                find_target_agent(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            find_target_agent(item)

                find_target_agent(eval_data)

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

    intents = []
    instruction_files = []
    for af in (agent_dir / "agents").glob("**/instruction.*"):
        if af.is_file():
            instruction_files.append(af)
            try:
                with open(af, "r", encoding="utf-8") as f:
                    content = f.read()
                agent_name = af.parent.name
                sections = re.findall(
                    r"<([a-zA-Z0-9_-]+)>(.*?)</\1>", content, re.DOTALL
                )

                def add_intent(quote_lines, cat_name):
                    q_text = " ".join(quote_lines).strip()
                    if len(q_text) > 10:
                        q_text = re.sub(r"^\d+[\.\)]\s*", "", q_text)
                        q_text = re.sub(r"^[\-\*]\s*", "", q_text)
                        q_text = q_text.strip()
                        directive_title = " ".join(q_text.split()[:5])
                        if len(directive_title) < len(q_text):
                            directive_title += "..."
                        intents.append(
                            {
                                "agent": agent_name,
                                "category": cat_name,
                                "directive": directive_title,
                                "quote": f'"{q_text[:60]}..."'
                                if len(q_text) > 60
                                else f'"{q_text}"',
                                "full_text": q_text,
                            }
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
                                add_intent(current_quote, category)
                                current_quote = [stripped]
                            else:
                                current_quote = [stripped]
                        else:
                            current_quote.append(stripped)
                    if current_quote:
                        add_intent(current_quote, category)

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
                                add_intent(current_quote, "Rules")
                                current_quote = [stripped]
                            else:
                                current_quote = [stripped]
                        else:
                            current_quote.append(stripped)
                    if current_quote:
                        add_intent(current_quote, "Rules")
            except Exception:
                pass

    covered_intents = []
    uncovered_intents = []
    for intent in intents:
        covered = False
        covering_evals = set()
        text_to_check = intent["full_text"].lower()

        match_tool = re.search(r"\{@TOOL[:\s]+([^}]+)\}", text_to_check)
        if match_tool:
            tool_name = match_tool.group(1).strip()
            if tool_name in called_tools:
                covered = True
                for ef in eval_files:
                    try:
                        with open(ef, "r", encoding="utf-8") as f:
                            if tool_name in f.read():
                                covered = True
                                eval_name = ef.stem
                                try:
                                    if ef.suffix == ".json":
                                        with open(
                                            ef, "r", encoding="utf-8"
                                        ) as f2:
                                            eval_j = json.load(f2)
                                            eval_name = (
                                                eval_j.get("displayName")
                                                or eval_j.get("name")
                                                or ef.name
                                            )
                                except Exception:
                                    pass
                                covering_evals.add(eval_name)
                    except Exception:
                        pass

        keywords = extract_keywords(text_to_check)
        for ef in eval_files:
            try:
                with open(ef, "r", encoding="utf-8") as f:
                    eval_content = f.read().lower()
                if any(kw in eval_content for kw in keywords if len(kw) > 4):
                    match_count = sum(
                        1
                        for kw in keywords
                        if len(kw) > 4 and kw in eval_content
                    )
                    if match_count >= 2 or (
                        len(keywords) == 1 and match_count == 1
                    ):
                        covered = True
                        eval_name = ef.stem
                        try:
                            if ef.suffix == ".json":
                                with open(ef, "r", encoding="utf-8") as f2:
                                    eval_j = json.load(f2)
                                    eval_name = (
                                        eval_j.get("displayName")
                                        or eval_j.get("name")
                                        or ef.name
                                    )
                        except Exception:
                            pass
                        covering_evals.add(eval_name)
            except Exception:
                pass

        intent["covered"] = "Yes" if covered else "No"
        intent["evals"] = (
            ", ".join(sorted(covering_evals)) if covering_evals else "None"
        )
        if covered:
            covered_intents.append(intent)
        else:
            uncovered_intents.append(intent)

    return (
        declared_transfers,
        covered_transfers,
        intents,
        covered_intents,
        instruction_files,
    )


def generate_report(
    output_file: Path,
    total_tools: Set[str],
    covered_tools: Set[str],
    phantom_tools_by_file: dict[Path, Set[str]],
    flat_segments: Dict[str, InstructionSegment],
    covered_segments: Set[str],
    uncovered_by_category: Dict[str, List[str]],
    eval_files: List[Path],
    declared_transfers: List[Tuple[str, str]],
    covered_transfers: Dict[Tuple[str, str], List[str]],
    intents: List[Dict[str, Any]],
    covered_intents: List[Dict[str, Any]],
    instruction_files: List[Path],
    agent_dir: Path,
) -> None:
    """Generates a clean, highly comprehensive Markdown coverage report."""
    uncovered_tools = total_tools - covered_tools
    tool_coverage_pct = (
        (len(covered_tools) / len(total_tools) * 100.0) if total_tools else 0.0
    )

    total_segments = len(intents)
    total_covered = len(covered_intents)
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

    for intent in intents:
        cat = "Static"
        if intent["category"] in ("Cond. Behavior", "Guardrails"):
            cat = "Conditional"
        elif intent["category"] in ("Conv. Flow",):
            cat = "Stateful"
        elif intent["category"] == "Tool":
            cat = "Tool"

        segment_counts[cat] += 1
        if intent["covered"] == "Yes":
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
        f"| **Tool Integrations** | {len(total_tools)} | {len(covered_tools)} | {tool_coverage_pct:.1f}% |"
    )
    report.append(
        f"| **Instruction Intents** | {total_segments} | {total_covered} | {overall_segment_pct:.1f}% |"
    )
    report.append(
        f"| **Agent Transfers** | {total_transfers} | {total_transfers_covered} | {transfer_coverage_pct:.1f}% |"
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
        f"| **Static segments** | Global Instructions, Persona, Voice, Guidelines | "
        f"{segment_counts['Static']} | {segment_covered['Static']} | {static_pct:.1f}% |"
    )

    stateful_pct = (
        (segment_covered["Stateful"] / segment_counts["Stateful"] * 100.0)
        if segment_counts["Stateful"]
        else 0.0
    )
    report.append(
        f"| **Stateful segments** | Context-dependent states (Slots)| "
        f"{segment_counts['Stateful']} | {segment_covered['Stateful']} | {stateful_pct:.1f}% |"
    )

    cond_pct = (
        (segment_covered["Conditional"] / segment_counts["Conditional"] * 100.0)
        if segment_counts["Conditional"]
        else 0.0
    )
    report.append(
        f"| **Conditional segments** | Branching logic Conditions, Routing | "
        f"{segment_counts['Conditional']} | {segment_covered['Conditional']} | {cond_pct:.1f}% |"
    )

    report.append("\n---\n")

    report.append("## Uncovered segments\n")
    has_uncovered = False
    for intent in intents:
        if intent["covered"] == "No":
            if not has_uncovered:
                report.append("### Uncovered Segments")
                has_uncovered = True
            report.append(f"*   `{intent['directive']}`")

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
        "| # | Agent | Category | Instruction Quote | Covered? | Covering Eval(s) |"
    )
    report.append(
        "|---|-------|----------|-------------------|----------|-------------------|"
    )
    for idx, intent in enumerate(intents, start=1):
        report.append(
            f"| {idx} | {intent['agent']} | {intent['category']} | {intent['quote']} | {intent['covered']} | {intent['evals']} |"
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


def calculate_segment_coverage(
    flat_segments: Dict[str, InstructionSegment],
    eval_texts: Set[str],
    called_tools: Set[str],
    agent_dir=None,
) -> Tuple[Set[str], Dict[str, List[str]]]:
    covered_segments = set()
    uncovered_by_category = {
        "Static": [],
        "Stateful": [],
        "Conditional": [],
        "Tool": [],
    }
    for segment_id, segment in flat_segments.items():
        covered = False
        if segment.category == "Tool":
            tool_name = segment.attributes.get("name", "")
            covered = tool_name in called_tools
        if covered:
            covered_segments.add(segment_id)
        else:
            cat = segment.category
            if cat in uncovered_by_category:
                uncovered_by_category[cat].append(segment_id)
    return covered_segments, uncovered_by_category


def run_coverage_analysis(
    agent_dir: Path,
) -> Tuple[
    Set[str],
    Set[str],
    Dict[Path, Set[str]],
    List[Dict[str, Any]],
    Dict[str, InstructionSegment],
    Set[str],
    Dict[str, List[str]],
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

    instruction_tree, flat_segments = analyze_instructions(agent_dir)

    covered_segments, uncovered_by_category = calculate_segment_coverage(
        flat_segments, eval_texts, called_tools, agent_dir
    )

    return (
        all_tools,
        covered_tools,
        phantom_tools_by_file,
        instruction_tree,
        flat_segments,
        covered_segments,
        uncovered_by_category,
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
    args = parser.parse_args()

    agent_dir = Path(args.agent_dir)
    output_file = Path(args.output_file)

    (
        all_tools,
        covered_tools,
        phantom_tools_by_file,
        instruction_tree,
        flat_segments,
        covered_segments,
        uncovered_by_category,
        eval_files,
        called_tools,
    ) = run_coverage_analysis(agent_dir)

    (
        declared_transfers,
        covered_transfers,
        intents,
        covered_intents,
        instruction_files,
    ) = extract_dynamic_coverage(agent_dir, eval_files, called_tools)

    print(f"Found {len(all_tools)} tool definition(s).")
    print(f"Found {len(flat_segments)} instruction segment(s).")
    print(f"Found {len(eval_files)} evaluation file(s).")

    if phantom_tools_by_file:
        print(
            "\n[WARNING] Detected tools that are referenced in evaluations but do not exist in the tools directory:"
        )
        for ef, phantoms in sorted(phantom_tools_by_file.items()):
            try:
                rel_path = ef.relative_to(agent_dir)
            except ValueError:
                rel_path = ef
            print(f"  - {rel_path}: {', '.join(sorted(phantoms))}")
        print(
            "Please verify if these tools were renamed, deleted, or misspelled.\n"
        )

    generate_report(
        output_file,
        all_tools,
        covered_tools,
        phantom_tools_by_file,
        flat_segments,
        covered_segments,
        uncovered_by_category,
        eval_files,
        declared_transfers,
        covered_transfers,
        intents,
        covered_intents,
        instruction_files,
        agent_dir,
    )


if __name__ == "__main__":
    main()
