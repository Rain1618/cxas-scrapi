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

"""Instruction-related evaluation coverage analysis functions."""

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml
from pydantic import BaseModel, Field

from cxas_scrapi.utils.gemini import GeminiGenerate
from utils import parse_instruction_content

class CategorizationResult(BaseModel):
    """Schema for LLM categorization of instruction segments."""
    category: str = Field(
        description="Category of the instruction: 'Functional Intent' or 'Behavioral Constraint'"
    )
    reasoning: str = Field(description="Reason for the categorization")


class SentimentAnalysisResult(BaseModel):
    """Schema for LLM sentiment analysis of user prompts."""
    has_behavioral_diversity: bool = Field(
        description="True if the test suite contains phrasing aimed at testing the personal, role or behaviour of the agent. False otherwise."
    )
    reasoning: str = Field(description="Reason for the decision")


def analyze_instructions(
    agent_dir: Path,
    gemini_client: GeminiGenerate = None,
) -> Tuple[List[Dict[str, Any]], List[Path]]:
    """Discovers and parses all instruction.txt and instruction.* files
    into instruction_segments."""
    instruction_segments = []
    instruction_files = []

    def parse_file(filepath: Path, agent_name: str):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            segments = parse_instruction_content(content, agent_name)
            instruction_segments.extend(segments)
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

    if gemini_client and instruction_segments:
        print(f"Categorizing {len(instruction_segments)} instruction segment(s) using LLM...")
        for segment in instruction_segments:
            prompt = f"""
            You are an expert in AI Agent design.
            Categorize the following AI Agent instruction into one of two categories:
            - 'Functional Intent': Explicit actions, API executions, or data retrievals (e.g., "Calculate the user's outstanding balance").
            - 'Behavioral Constraint': Quality, tone, persona, or safety guardrails (e.g., "Be nice and welcoming").

            Instruction:
            <INSTRUCTION>
            {segment['full_text']}
            </INSTRUCTION>

            Analyze the instruction and determine the best category.
            """
            try:
                llm_response = gemini_client.generate(
                    prompt=prompt,
                    response_mime_type="application/json",
                    response_schema=CategorizationResult,
                    temperature=0.0,
                )
                if llm_response:
                    cat = getattr(llm_response, "category", "Functional Intent")
                    if isinstance(llm_response, dict):
                        cat = llm_response.get("category", "Functional Intent")

                    # Normalize category to match the requested names
                    if "functional" in cat.lower():
                        segment["category"] = "Functional Intent"
                    elif "behavioral" in cat.lower() or "persona" in cat.lower() or "constraint" in cat.lower():
                        segment["category"] = "Behavioral Constraint"
                    else:
                        segment["category"] = cat # Fallback to LLM response if it returns something else
            except Exception as e:
                print(f"Warning: LLM categorization failed for segment '{segment['directive']}': {e}")
                # Keep original category (which might be from XML tag or "Rules") or default
                if segment["category"] not in ["Functional Intent", "Behavioral Constraint"]:
                     # Try to infer from current category
                     if "rule" in segment["category"].lower() or "persona" in segment["category"].lower():
                         segment["category"] = "Behavioral Constraint"
                     else:
                         segment["category"] = "Functional Intent"

    return instruction_segments, instruction_files


def extract_all_user_prompts(eval_files: List[Path]) -> List[str]:
    """Extracts all user prompts from evaluation files."""
    user_prompts = []
    for ef in eval_files:
        try:
            if ef.suffix in (".yaml", ".yml"):
                with open(ef, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if not data:
                    continue
                if "conversations" in data:
                    for conv in data.get("conversations", []):
                        for turn in conv.get("turns", []):
                            user = turn.get("user", "")
                            if user:
                                user_prompts.append(user)
            elif ef.suffix == ".json":
                with open(ef, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    golden = data.get("golden", {})
                    for turn in golden.get("turns", []):
                        for step in turn.get("steps", []):
                            if "userInput" in step:
                                text = step["userInput"].get("text", "")
                                if text:
                                    user_prompts.append(text)
        except Exception as e:
            print(f"Warning: Failed to extract user prompts from {ef}: {e}")
    return user_prompts


def extract_instruction_coverage(
    instruction_segments: List[Dict[str, Any]],
    eval_files: List[Path],
    called_tools: Set[str],
    gemini_client: GeminiGenerate = None,
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

    behavioral_constraints = [
        s["full_text"]
        for s in instruction_segments
        if s.get("category") == "Behavioral Constraint"
    ]
    persona_text = "\n".join(f"- {c}" for c in behavioral_constraints)

    if not persona_text:
        persona_text = "Standard professional, helpful, and polite AI Agent persona."

    user_prompts = extract_all_user_prompts(eval_files)
    has_behavioral_diversity = False

    if gemini_client and user_prompts:
        print(f"Analyzing {len(user_prompts)} user prompt(s) for behavioral diversity (dynamic persona)...")
        # Sample prompts if too many to avoid context window blowup
        sampled_prompts = user_prompts[:100] # Adjust as needed
        prompts_text = "\n".join(f"- {p}" for p in sampled_prompts)

        prompt = f"""
        You are an expert LLM as a Judge determining evaluation coverage for an AI Agent.
        Analyze the following set of user prompts and/or evaluation chunks used in AI Agent evaluation tests.

        The AI Agent is configured with the following persona and behavioral constraints:
        <PERSONA_AND_BEHAVIORAL_CONSTRAINTS>
        {persona_text}
        </PERSONA_AND_BEHAVIORAL_CONSTRAINTS>

        Determine if the test suite gives the AI Agent a meaningful opportunity to demonstrate that it follows its defined persona and behavioral constraints (e.g., in Simulation Evals, Golden Evals, or conversational steps).
        - If the evaluation suite provides scenarios, conversations, or goals where the Agent's persona can naturally be expressed or validated (e.g., polite phrasing, helpfulness, professional tone, or simple adherence to persona rules), respond with `true` in `has_behavioral_diversity`.
        - ONLY respond with `false` if the evaluations completely lack any user inputs, or if the defined persona includes highly specific, strict, or critical behavioral boundaries and guardrails (e.g., "never discuss pricing", "do not mention competitor X") that are completely untested or unexplored in the provided evaluation chunks.

        User Prompts / Scenarios to Analyze:
        {prompts_text}
        """
        try:
            llm_response = gemini_client.generate(
                prompt=prompt,
                response_mime_type="application/json",
                response_schema=SentimentAnalysisResult,
                temperature=0.0,
                model_name="gemini-2.5-flash",
            )
            if llm_response:
                has_behavioral_diversity = getattr(llm_response, "has_behavioral_diversity", False)
                if isinstance(llm_response, dict):
                    has_behavioral_diversity = llm_response.get("has_behavioral_diversity", False)
                print(f"Behavioral Diversity Analysis Result: {has_behavioral_diversity}")
        except Exception as e:
            print(f"Warning: Dynamic persona sentiment analysis failed: {e}")
            has_behavioral_diversity = False # Default to False on failure to be conservative
    elif not user_prompts:
        print("No explicit user prompts found for behavioral diversity analysis.")
        has_behavioral_diversity = False

    if not gemini_client:
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
                Determine if ANY of these evaluation chunks explicitly test or provide a natural opportunity to demonstrate that the Agent follows the provided Agent Instruction.
                - For general persona, tone, or behavioral constraints (e.g., "be polite", "sound professional", "be patient"), if the evaluation chunk allows the agent to carry out a natural conversation or achieve its goal under these guidelines, consider it covered.
                - Only mark as uncovered if the instruction contains a highly specific rule or guardrail that is explicitly not triggered, tested, or challenged by the evaluation chunk.
                Answer true in `is_covered` if at least one evaluation chunk covers or allows natural demonstration of the instruction, and identify the FIRST covering chunk's index (0-based) in `covering_chunk_index`.
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

        if covered:
            instruction_segment["covered"] = "Yes"
            covered_instruction_segments.append(instruction_segment)
        else:
            instruction_segment["covered"] = "No"
            uncovered_instruction_segments.append(instruction_segment)

        instruction_segment["evals"] = (
            ", ".join(sorted(covering_evals)) if covering_evals else "None"
        )

    return instruction_segments, covered_instruction_segments
