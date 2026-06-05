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

import asyncio
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from pydantic import BaseModel, Field
from utils import cosine_similarity

from cxas_scrapi.utils.gemini import GeminiGenerate


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


class ConsolidatedInstruction(BaseModel):
    """Schema for a single consolidated/filtered instruction segment."""

    directive: str = Field(
        description="A brief 3-5 word title/summary of the instruction segment."
    )
    full_text: str = Field(
        description="The complete, detailed instruction text. Keep specific parameters, tool names, and validation rules intact."
    )
    category: str = Field(
        description="The category: 'Functional Intent' or 'Behavioral Constraint'."
    )
    reasoning: str = Field(
        description="A brief explanation of why this segment is kept or how it was consolidated."
    )


class ConsolidationResult(BaseModel):
    """Schema for the LLM instruction segment consolidation result."""

    instructions: List[ConsolidatedInstruction] = Field(
        description="The list of refined, consolidated instruction segments that actually need to be tested."
    )


async def consolidate_instruction_segments_with_llm(
    instruction_segments: List[Dict[str, Any]],
    gemini_client: GeminiGenerate,
    errors: List[str] = None,
) -> List[Dict[str, Any]]:
    """Uses a pro Gemini model to consolidate and extract testable instruction segments."""
    if not gemini_client or not instruction_segments:
        return instruction_segments

    segments_by_agent = defaultdict(list)
    for s in instruction_segments:
        segments_by_agent[s["agent"]].append(s)

    consolidated_segments = []
    sem = asyncio.Semaphore(5)

    async def process_agent(agent_name: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        async with sem:
            print(
            f"Consolidating {len(segments)} raw instruction segment(s) "
            f"for agent '{agent_name}' using Pro LLM..."
        )

        # Format raw segments for the prompt
        raw_segments_text = ""
        for idx, s in enumerate(segments):
            raw_segments_text += (
                f"--- RAW SEGMENT {idx} ---\n"
                f"Category: {s.get('category', 'Rules')}\n"
                f"Text: {s['full_text']}\n\n"
            )

        prompt = f"""
        You are an expert test coverage engineer and Conversational AI designer.
        Your task is to review, refine, and consolidate the following list of raw instruction segments extracted from the GECX agent '{agent_name}'.

        Please perform these steps carefully:
        1. **Merge Redundancies**: Combine instruction segments that are highly similar, redundant, or represent variations of the same core guideline/behavior into a single comprehensive segment.
        2. **Filter Non-Testables**: Remove instructions that are non-testable, conversational filler, general formatting rules not related to logic, or boilerplate greetings (e.g., "Greet the user nicely", "Always say hello"). Keep instructions that describe specific functional intents, API/tool execution business logic, conditional routing logic, strict validation constraints, and distinctive behavioral or safety guardrails.
        3. **Categorize**: Ensure each final consolidated segment is classified as either:
           - 'Functional Intent': Explicit actions, tool executions, API logic, or data retrieval steps.
           - 'Behavioral Constraint': Tone, safety guardrails, active listening techniques, or conversational persona rules.
        4. **Format Output**: Ensure every segment has a 3-5 word directive title, the complete consolidated full_text, the category, and a brief reasoning explanation.

        Raw Instruction Segments:
        {raw_segments_text}
        """

        try:
            # Requesting gemini-2.5-pro explicitly as the "pro agent"
            # for complex consolidation
            response = await gemini_client.generate_async(
                prompt=prompt,
                model_name="gemini-2.5-pro",
                response_mime_type="application/json",
                response_schema=ConsolidationResult,
                temperature=0.0,
            )

            if response and hasattr(response, "instructions"):
                instructions_list = response.instructions
            elif (
                response
                and isinstance(response, dict)
                and "instructions" in response
            ):
                instructions_list = response["instructions"]
            else:
                instructions_list = []

            agent_consolidated = []
            for inst in instructions_list:
                if isinstance(inst, dict):
                    dir_val = inst.get("directive", "")
                    text_val = inst.get("full_text", "")
                    cat_val = inst.get("category", "Functional Intent")
                else:
                    dir_val = getattr(inst, "directive", "")
                    text_val = getattr(inst, "full_text", "")
                    cat_val = getattr(inst, "category", "Functional Intent")

                if len(text_val) > 10:
                    q_text = text_val.strip()
                    agent_consolidated.append(
                        {
                            "agent": agent_name,
                            "category": cat_val,
                            "directive": dir_val,
                            "quote": f'"{q_text[:200]}..."'
                            if len(q_text) > 200
                            else f'"{q_text}"',
                            "full_text": q_text,
                        }
                    )
            return agent_consolidated
        except Exception as e:
            err_msg = f"Pro LLM consolidation failed for agent '{agent_name}': {e}"
            print(f"Warning: {err_msg}")
            if errors is not None:
                errors.append(err_msg)
            # Fallback: use original segments if LLM fails
            return segments

    tasks = [process_agent(agent_name, segments) for agent_name, segments in segments_by_agent.items()]
    results = await asyncio.gather(*tasks)
    for res in results:
        consolidated_segments.extend(res)

    return consolidated_segments


async def analyze_instruction_categories(
    instruction_segments: List[Dict[str, Any]],
    gemini_client: GeminiGenerate = None,
    errors: List[str] = None,
) -> List[Dict[str, Any]]:
    """Runs LLM classification on instruction segments to categorize them."""
    if not gemini_client or not instruction_segments:
        return instruction_segments

    print(
        f"Categorizing {len(instruction_segments)} instruction segment(s) using LLM..."
    )

    sem = asyncio.Semaphore(5)

    async def process_segment(segment: Dict[str, Any]):
        async with sem:
            prompt = f"""
        Categorize the following AI Agent instruction into one of two categories:
        - 'Functional Intent': Explicit actions, API executions, or data retrievals (e.g., "Calculate the user's outstanding balance").
        - 'Behavioral Constraint': Quality, tone, persona, or safety guardrails (e.g., "Be nice and welcoming").

        Instruction:
        <INSTRUCTION>
        {segment["full_text"]}
        </INSTRUCTION>

        Analyze the instruction and determine the best category.
        """
        try:
            llm_response = await gemini_client.generate_async(
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
                elif (
                    "behavioral" in cat.lower()
                    or "persona" in cat.lower()
                    or "constraint" in cat.lower()
                ):
                    segment["category"] = "Behavioral Constraint"
                else:
                    segment["category"] = (
                        cat  # Fallback to LLM response if it returns something else
                    )
        except Exception as e:
            err_msg = f"LLM categorization failed for segment '{segment['directive']}': {e}"
            print(f"Warning: {err_msg}")
            if errors is not None:
                errors.append(err_msg)
            # Keep original category (which might be from XML tag or "Rules") or default
            if segment.get("category") not in [
                "Functional Intent",
                "Behavioral Constraint",
            ]:
                # Try to infer from current category
                orig_cat = segment.get("category", "Rules")
                if (
                    "rule" in orig_cat.lower()
                    or "persona" in orig_cat.lower()
                ):
                    segment["category"] = "Behavioral Constraint"
                else:
                    segment["category"] = "Functional Intent"

    tasks = [process_segment(seg) for seg in instruction_segments]
    await asyncio.gather(*tasks)
    return instruction_segments


async def extract_instruction_coverage(
    instruction_segments: List[Dict[str, Any]],
    eval_chunks: List[Dict[str, Any]],
    called_tools: Set[str],
    gemini_client: GeminiGenerate = None,
    errors: List[str] = None,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Uses Vector Embeddings and LLM-as-a-judge to determine instruction
    segment coverage against pre-computed eval chunks."""

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

    # Reduce batch size and limit concurrency to prevent backend overload
    emb_sem = asyncio.Semaphore(3)

    async def batch_generate_embeddings_async(
        texts: List[str], batch_size: int = 100
    ) -> List[Any]:
        if not texts:
            return []

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        async def get_batch_embeddings(batch):
            async with emb_sem:
                try:
                    # Adding a small sleep to space out requests
                    await asyncio.sleep(1)
                    return await asyncio.to_thread(
                        gemini_client.generate_embeddings, contents=batch
                    )
                except Exception as e:
                    err_msg = f"Failed to generate embeddings for batch: {e}"
                    print(f"Warning: {err_msg}")
                    if errors is not None:
                        errors.append(err_msg)
                    return [None] * len(batch)

        tasks = [get_batch_embeddings(b) for b in batches]
        results = await asyncio.gather(*tasks)

        embeddings = []
        for res in results:
            embeddings.extend(res)
        return embeddings

    instruction_segment_embeddings = []
    chunk_embeddings = []

    if instruction_segments_texts:
        print(
            f"Generating embeddings for {len(instruction_segments_texts)} "
            "instruction segment(s)..."
        )
        instruction_segment_embeddings = await batch_generate_embeddings_async(
            instruction_segments_texts
        )

    if chunk_texts:
        print(f"Generating embeddings for {len(chunk_texts)} eval chunk(s)...")
        chunk_embeddings = await batch_generate_embeddings_async(chunk_texts)

    judge_sem = asyncio.Semaphore(5)

    async def run_llm_judge(
        instruction_text: str,
        candidate_chunks: List[Dict[str, Any]],
        idx: int,
    ) -> Tuple[int, bool, int]:
        async with judge_sem:
            chunks_formatted_text = ""
        for c_idx, c in enumerate(candidate_chunks):
            chunks_formatted_text += (
                f"\n--- CANDIDATE CHUNK {c_idx} ---\n{c['text']}\n"
            )

        prompt = f"""
        You are an expert LLM as a Judge determining evaluation coverage
        for an AI Agent.

        Agent Instruction to Test:
        <INSTRUCTION>
        {instruction_text}
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
            llm_response = await gemini_client.generate_async(
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

                return idx, is_cov, c_idx
        except Exception as e:
            err_msg = f"LLM call failed for instruction segment {idx}: {e}"
            print(err_msg)
            if errors is not None:
                errors.append(err_msg)

        return idx, False, -1

    # Initialize coverage states
    segment_states = []
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
                for chunk in eval_chunks:
                    if tool_name in chunk["text"]:
                        covering_evals.add(chunk["eval_name"])

        segment_states.append({
            "covered": covered,
            "covering_evals": covering_evals,
            "candidate_chunks": []
        })

    # 2. Prepare tasks for LLM Judge where needed
    llm_tasks = []
    for i, instruction_segment in enumerate(instruction_segments):
        if segment_states[i]["covered"]:
            continue

        if (
            i < len(instruction_segment_embeddings)
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

            # Get top 5 most relevant chunks
            similarities.sort(reverse=True, key=lambda x: x[0])
            top_candidates = similarities[:5]

            candidate_chunks = [
                eval_chunks[idx] for sim, idx in top_candidates if sim > 0.0
            ]

            if candidate_chunks:
                segment_states[i]["candidate_chunks"] = candidate_chunks
                llm_tasks.append(
                    run_llm_judge(
                        instruction_segment["full_text"],
                        candidate_chunks,
                        i
                    )
                )

    # 3. Execute LLM calls concurrently
    if llm_tasks:
        print(f"Running {len(llm_tasks)} instruction coverage LLM-as-a-judge calls in parallel...")
        llm_results = await asyncio.gather(*llm_tasks)

        for idx, is_cov, c_idx in llm_results:
            if is_cov and 0 <= c_idx < len(segment_states[idx]["candidate_chunks"]):
                segment_states[idx]["covered"] = True
                covering_chunk = segment_states[idx]["candidate_chunks"][c_idx]
                segment_states[idx]["covering_evals"].add(covering_chunk["eval_name"])

    # 4. Finalize segments
    covered_instruction_segments = []
    for i, instruction_segment in enumerate(instruction_segments):
        state = segment_states[i]
        if state["covered"]:
            instruction_segment["covered"] = "Yes"
            covered_instruction_segments.append(instruction_segment)
        else:
            instruction_segment["covered"] = "No"

        instruction_segment["evals"] = (
            ", ".join(sorted(state["covering_evals"])) if state["covering_evals"] else "None"
        )

    return instruction_segments, covered_instruction_segments


class DesiredTransfersResult(BaseModel):
    """Schema for the LLM evaluation of desired agent transfers."""

    desired_target_agents: List[str] = Field(
        description="The exact names of the target agents that this agent could potentially transfer to."
    )
    reasoning: str = Field(
        description="A brief explanation of how these targets were identified."
    )


async def determine_desired_transfers_with_llm(
    agent_directories: Dict[str, Path],
    declared_transfers: List[Tuple[str, str]],
    gemini_client: GeminiGenerate,
    errors: List[str] = None,
) -> Set[Tuple[str, str]]:
    """Uses LLM to determine which of the declared transfers are actually desired by the agent."""
    if not gemini_client or not declared_transfers:
        return set()

    desired_transfers = set()

    # Group by from_agent
    outbound_transfers = defaultdict(list)
    for from_a, to_a in declared_transfers:
        outbound_transfers[from_a].append(to_a)

    sem = asyncio.Semaphore(5)

    async def process_agent(agent_name: str, possible_targets: List[str]):
        async with sem:
            if agent_name not in agent_directories:
                return

        agent_dir = agent_directories[agent_name]

        # Read all relevant files for this agent
        files_to_check = []
        files_to_check.extend(agent_dir.glob("instruction.*"))
        files_to_check.extend(agent_dir.glob("*.json"))
        files_to_check.extend(agent_dir.glob("*.yaml"))
        files_to_check.extend(agent_dir.glob("*.yml"))
        files_to_check.extend(agent_dir.glob("**/*callbacks*/*/python_code.py"))

        content_parts = []
        for f in files_to_check:
            if not f.is_file():
                continue
            try:
                text = f.read_text(encoding="utf-8")
                # Try to abbreviate if too long (e.g. limit to 10k chars per file to avoid context bloat)
                if len(text) > 10000:
                    text = text[:10000] + "... (truncated)"
                content_parts.append(f"--- FILE: {f.name} ---\n{text}\n")
            except Exception:
                pass

        agent_files_content = "\n".join(content_parts)
        if not agent_files_content.strip():
            return

        print(f"Determining desired transfers for '{agent_name}' with LLM...")

        prompt = f"""
        You are an expert analyzing a GECX conversational agent's configuration and logic.

        Agent Name: {agent_name}
        Theoretically Possible Target Agents: {json.dumps(possible_targets)}

        Based on the agent's instructions, configuration, and callback logic provided below, determine which of the 'Theoretically Possible Target Agents' this agent actually intends to transfer to.
        A transfer might be explicitly mentioned in the instructions (e.g., "transfer to the billing agent" or a tool call like `set_active_flow` with flow="billing") or within the callback logic (e.g., `Part.from_agent_transfer`).
        Only include targets that have clear evidence of being an intended destination.

        Agent Files Content:
        {agent_files_content}
        """

        try:
            llm_response = await gemini_client.generate_async(
                prompt=prompt,
                response_mime_type="application/json",
                response_schema=DesiredTransfersResult,
                temperature=0.0,
            )

            if llm_response:
                targets = getattr(llm_response, "desired_target_agents", [])
                if isinstance(llm_response, dict):
                    targets = llm_response.get("desired_target_agents", [])

                for t in targets:
                    # Validate that the returned target is one of the possible targets
                    for pt in possible_targets:
                        if t.lower() == pt.lower():
                            desired_transfers.add((agent_name, pt))
                            break
        except Exception as e:
            err_msg = f"LLM desired transfer extraction failed for '{agent_name}': {e}"
            print(f"Warning: {err_msg}")
            if errors is not None:
                errors.append(err_msg)

    tasks = [process_agent(agent_name, targets) for agent_name, targets in outbound_transfers.items()]
    await asyncio.gather(*tasks)

    return desired_transfers
