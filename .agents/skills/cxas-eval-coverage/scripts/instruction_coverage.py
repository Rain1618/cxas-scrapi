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
from typing import Any, Dict, List, Set, Tuple

from pydantic import BaseModel, Field

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

def analyze_instruction_categories(
    instruction_segments: List[Dict[str, Any]],
    gemini_client: GeminiGenerate = None,
) -> List[Dict[str, Any]]:
    """Runs LLM classification on instruction segments to categorize them."""
    if gemini_client and instruction_segments:
        print(f"Categorizing {len(instruction_segments)} instruction segment(s) using LLM...")
        for segment in instruction_segments:
            prompt = f"""
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

    return instruction_segments


def extract_instruction_coverage(
    instruction_segments: List[Dict[str, Any]],
    eval_chunks: List[Dict[str, Any]],
    called_tools: Set[str],
    gemini_client: GeminiGenerate = None,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Uses Vector Embeddings and LLM-as-a-judge to determine instruction
    segment coverage against pre-computed eval chunks."""

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

    covered_instruction_segments = []

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
                for chunk in eval_chunks:
                    if tool_name in chunk["text"]:
                        covering_evals.add(chunk["eval_name"])

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

        instruction_segment["evals"] = (
            ", ".join(sorted(covering_evals)) if covering_evals else "None"
        )

    return instruction_segments, covered_instruction_segments
