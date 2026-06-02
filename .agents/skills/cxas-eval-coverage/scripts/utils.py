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

"""Utility functions for GECX evaluation coverage script."""

import re
from typing import Any, Dict, List


def parse_instruction_content(
    content: str, agent_name: str
) -> List[Dict[str, Any]]:
    """Parses instruction file content and splits it into structured segments.

    Supports both XML-tagged sections (e.g., <Rules>...) and raw files (fallback to 'Rules').
    """
    instruction_segments = []

    def add_instruction_segment(
        quote_lines: List[str], cat_name: str, a_name: str
    ):
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
                    "quote": f'"{q_text[:200]}..."'
                    if len(q_text) > 200
                    else f'"{q_text}"',
                    "full_text": q_text,
                }
            )

    def chunk_lines_into_segments(lines_list: List[str], cat_name: str):
        current_quote = []
        for line in lines_list:
            stripped = line.strip()
            if not stripped:
                continue
            if (
                re.match(r"^\d+[\.\)]\s*", stripped)
                or stripped.startswith("-")
                or stripped.startswith("*")
            ):
                if current_quote:
                    add_instruction_segment(current_quote, cat_name, agent_name)
                    current_quote = [stripped]
                else:
                    current_quote = [stripped]
            else:
                current_quote.append(stripped)
        if current_quote:
            add_instruction_segment(current_quote, cat_name, agent_name)

    sections = re.findall(r"<([a-zA-Z0-9_-]+)>(.*?)</\1>", content, re.DOTALL)

    for tag, text in sections:
        category = tag.replace("_", " ").title()
        chunk_lines_into_segments(text.split("\n"), category)

    if not sections:
        chunk_lines_into_segments(content.split("\n"), "Rules")

    return instruction_segments
