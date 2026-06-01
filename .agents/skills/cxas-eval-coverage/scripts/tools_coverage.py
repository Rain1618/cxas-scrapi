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

"""Tools-related evaluation coverage analysis functions."""

import json
import re
from pathlib import Path
from typing import Set

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
