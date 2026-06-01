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

"""Data ingestion module for GECX evaluation coverage analyzer."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

from utils import parse_instruction_content


@dataclass
class AgentProjectData:
    """A unified data model representing the fully ingested GECX agent project."""
    agent_dir: Path
    all_tools: Set[str] = field(default_factory=set)
    eval_files: List[Path] = field(default_factory=list)

    # Aggregated tool coverage metrics (from ingestion)
    called_tools: Set[str] = field(default_factory=set)
    covered_tools: Set[str] = field(default_factory=set)
    phantom_tools_by_file: Dict[Path, Set[str]] = field(default_factory=dict)

    # Sub-agent transitions/transfers
    declared_transfers: List[Tuple[str, str]] = field(default_factory=list)
    covered_transfers: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)

    # Pre-computed evaluation chunks for instruction similarity judge
    eval_chunks: List[Dict[str, Any]] = field(default_factory=list)

    # Ingested instruction files and raw segments
    instruction_files: List[Path] = field(default_factory=list)
    instruction_segments: List[Dict[str, Any]] = field(default_factory=list)


def find_tools_local(tools_dir: Path) -> Set[str]:
    """Finds all declared tool names in the tools directory."""
    tools = set()
    if not tools_dir.exists():
        return tools

    for p in tools_dir.glob("**/*"):
        if p.suffix in (".json", ".yaml", ".yml"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    content = (
                        yaml.safe_load(f)
                        if p.suffix in (".yaml", ".yml")
                        else json.load(f)
                    )

                tool_name = p.stem
                if isinstance(content, dict):
                    if "displayName" in content:
                        tool_name = content["displayName"]
                    elif "name" in content:
                        name_val = content["name"]
                        if not re.match(r"^[0-9a-fA-F-]{36}$", str(name_val)):
                            tool_name = name_val
                tools.add(tool_name)
            except Exception:
                tools.add(p.stem)
    return tools


def ingest_agent_project(agent_dir: Path) -> AgentProjectData:
    """Ingests and parses all agent files in a single pass, building AgentProjectData."""
    data = AgentProjectData(agent_dir=agent_dir)

    # 1. Find declared tools
    tools_dir = agent_dir / "tools"
    data.all_tools = find_tools_local(tools_dir)

    # 2. Find all evaluation files
    eval_dir = agent_dir / "evaluations"
    eval_dataset_dir = agent_dir / "evaluationDatasets"
    for d in (eval_dir, eval_dataset_dir):
        if d.exists():
            for p in d.glob("**/*"):
                if p.is_file() and p.suffix in (".json", ".yaml", ".yml"):
                    data.eval_files.append(p)

    # 3. Discover declared Agent Transfers from Agents config
    agent_files = list((agent_dir / "agents").glob("**/*.json"))
    agents = {}
    root_agents = set()
    for af in agent_files:
        try:
            with open(af, "r", encoding="utf-8") as f:
                agent_data = json.load(f)
            display_name = agent_data.get("displayName")
            agents[display_name] = agent_data
            root_agents.add(display_name)
        except Exception:
            pass

    for display_name, agent_data in agents.items():
        children = agent_data.get("childAgents", [])
        for child in children:
            root_agents.discard(child)
            data.declared_transfers.append((display_name, child))
            for c2 in children:
                if child != c2:
                    data.declared_transfers.append((child, c2))

    default_root_agent = next(iter(root_agents)) if root_agents else None

    # 4. Ingest all evaluations in a single, unified file-reading pass
    for ef in data.eval_files:
        file_tools = set()
        try:
            if ef.suffix == ".json":
                with open(ef, "r", encoding="utf-8") as f:
                    eval_content = json.load(f)

                if isinstance(eval_content, dict):
                    eval_name = eval_content.get("displayName") or eval_content.get("name") or ef.stem
                    golden = eval_content.get("golden", {})
                    turns = golden.get("turns", [])
                    for turn_idx, turn in enumerate(turns):
                        steps = turn.get("steps", [])
                        turn_text = []
                        for step in steps:
                            if "userInput" in step:
                                turn_text.append(f"User: {step['userInput'].get('text', '')}")

                            expectation = step.get("expectation")
                            if isinstance(expectation, dict):
                                # Track tool calls
                                tool_call = expectation.get("toolCall")
                                if isinstance(tool_call, dict) and "tool" in tool_call:
                                    file_tools.add(tool_call["tool"])

                                # Compile expectation criteria for vector chunks
                                if "note" in expectation:
                                    turn_text.append(f"Expectation Note: {expectation['note']}")
                                if "agentTransfer" in expectation:
                                    target_ag = expectation["agentTransfer"].get("targetAgent", "")
                                    turn_text.append(f"Expects Transfer to: {target_ag}")
                                if "toolCall" in expectation:
                                    turn_text.append(f"Expects Tool Call: {expectation['toolCall'].get('tool', '')}")
                                if "updatedVariables" in expectation:
                                    turn_text.append(f"Expects Updated Variables: {json.dumps(expectation['updatedVariables'])}")

                        if turn_text:
                            data.eval_chunks.append({
                                "text": f"Native Eval: {ef.stem} (Turn {turn_idx})\n" + "\n".join(turn_text),
                                "eval_name": eval_name,
                                "file_name": ef.name
                            })

                    # Track agent transfers
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

                    find_target_agent(eval_content, target_agents)

                    current_agent = default_root_agent
                    for target in target_agents:
                        if current_agent and target:
                            edge = (current_agent, target)
                            if edge not in data.covered_transfers:
                                data.covered_transfers[edge] = []
                            if eval_name not in data.covered_transfers[edge]:
                                data.covered_transfers[edge].append(eval_name)
                            current_agent = target

            elif ef.suffix in (".yaml", ".yml"):
                with open(ef, "r", encoding="utf-8") as f:
                    eval_content = yaml.safe_load(f)
                if not eval_content:
                    continue

                # SCRAPI Golden Evals
                if "conversations" in eval_content:
                    for conv in eval_content["conversations"]:
                        c_name = conv.get("conversation", "Unnamed")
                        tags = conv.get("tags", [])

                        turns_text = []
                        for turn in conv.get("turns", []):
                            user = turn.get("user", "")
                            agent = turn.get("agent", "")
                            turn_str = f"User: {user}\nAgent: {agent}"

                            for tool_call in turn.get("tool_calls", []):
                                if isinstance(tool_call, dict) and "action" in tool_call:
                                    file_tools.add(tool_call["action"])

                            if "tool_calls" in turn:
                                turn_str += f"\nTool Calls: {json.dumps(turn['tool_calls'])}"
                            turns_text.append(turn_str)

                        if turns_text:
                            data.eval_chunks.append({
                                "text": f"Conversation: {c_name}\nTags: {', '.join(tags)}\n" + "\n".join(turns_text),
                                "eval_name": c_name or ef.stem,
                                "file_name": ef.name
                            })

                        expectations = conv.get("expectations", [])
                        if expectations:
                            data.eval_chunks.append({
                                "text": f"Conversation: {c_name}\nExpectations:\n" + "\n".join(f"- {exp}" for exp in expectations),
                                "eval_name": c_name or ef.stem,
                                "file_name": ef.name
                            })

                # SCRAPI Simulation Evals
                elif "evals" in eval_content:
                    for eval_item in eval_content["evals"]:
                        e_name = eval_item.get("name", "Unnamed")
                        tags = eval_item.get("tags", [])

                        steps_text = []
                        for step in eval_item.get("steps", []):
                            goal = step.get("goal", "")
                            success = step.get("success_criteria", "")
                            guide = step.get("response_guide", "")
                            steps_text.append(f"Goal: {goal}\nSuccess Criteria: {success}\nResponse Guide: {guide}")

                        if steps_text:
                            data.eval_chunks.append({
                                "text": f"Simulation Eval: {e_name}\nTags: {', '.join(tags)}\n" + "\n".join(steps_text),
                                "eval_name": e_name or ef.stem,
                                "file_name": ef.name
                            })

                        expectations = eval_item.get("expectations", [])
                        if expectations:
                            data.eval_chunks.append({
                                "text": f"Simulation Eval: {e_name}\nExpectations:\n" + "\n".join(f"- {exp}" for exp in expectations),
                                "eval_name": e_name or ef.stem,
                                "file_name": ef.name
                            })

                        # Extract tools from expectations & success criteria
                        text_to_scan = []
                        text_to_scan.extend(expectations)
                        for step in eval_item.get("steps", []):
                            if "success_criteria" in step:
                                text_to_scan.append(step["success_criteria"])
                            if "goal" in step:
                                text_to_scan.append(step["goal"])

                        for text in text_to_scan:
                            if not isinstance(text, str):
                                continue
                            for tool in data.all_tools:
                                if re.search(rf"\b{re.escape(tool)}\b", text, re.IGNORECASE):
                                    file_tools.add(tool)

        except Exception as e:
            print(f"Warning: Failed to ingest evaluation file {ef}: {e}")

        # Check for phantom tools & compile coverage
        phantoms = file_tools - data.all_tools - {"end_session"}
        if phantoms:
            data.phantom_tools_by_file[ef] = phantoms

        data.covered_tools.update(file_tools & data.all_tools)
        data.called_tools.update(file_tools)

    # 5. Ingest all instruction files recursively
    agents_dir = agent_dir / "agents"
    def parse_instruction_file(filepath: Path, agent_name: str):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            segments = parse_instruction_content(content, agent_name)
            data.instruction_segments.extend(segments)
        except Exception as e:
            print(f"Warning: Failed to parse instructions {filepath}: {e}")

    if agents_dir.exists() and agents_dir.is_dir():
        for p in agents_dir.glob("**/instruction.*"):
            if p.is_file():
                data.instruction_files.append(p)
                parse_instruction_file(p, p.parent.name)

    p = agent_dir / "global_instruction.txt"
    if p.is_file():
        data.instruction_files.append(p)
        parse_instruction_file(p, "Global")

    return data
