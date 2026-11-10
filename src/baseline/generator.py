"""Local Qwen generation client using Hugging Face Transformers on CPU."""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch

# Set model cache to D: drive
HF_STORAGE_DIR = "runtime/research/models"
os.environ.setdefault("HF_HOME", HF_STORAGE_DIR)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.core.models import RetrievedChunk


SYSTEM_PROMPT = """You are an enterprise assistant for Sahyadri Software Services.
Answer the user's question accurately using ONLY the retrieved document evidence provided below.
Rules:
1. Ground your answer strictly in the provided clauses. Do NOT assume or invent external facts.
2. If the retrieved evidence contains conflicting or incompatible statements for the same scope without an authoritative resolution, explicitly state that there is an unresolved conflict between the documents and mention both values.
3. If an announced policy has an effective date in the future relative to the question date, state what is currently in effect today and clearly mention that the announced change starts in the future.
4. If asked about a past period, answer with the rule that governed at that historical date.
5. Provide your response strictly in valid JSON format:
{
  "answer": "<direct factual answer>",
  "cited_clauses": ["<document_id#clause_id>", ...],
  "reasoning": "<brief explanation citing dates and authorities>"
}
"""


class LocalQwenGenerator:
    """Generation client running Qwen2.5 on CPU with strict temperature=0."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        cache_dir: str = HF_STORAGE_DIR,
        device: str = "cpu",
        torch_dtype: str = "float32",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        mock_mode: bool = False
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype, torch.float32)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.mock_mode = mock_mode

        self._tokenizer = None
        self._model = None

    def _load_model(self):
        if self._model is None and not self.mock_mode:
            print(f"[LLM] Loading {self.model_name} on {self.device} (cache: {self.cache_dir})...")
            torch.set_num_threads(8)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir)
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                dtype=self.torch_dtype,
                low_cpu_mem_usage=True
            ).to(self.device)

    def format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Formats retrieved chunks into evidence blocks for the prompt."""
        context_blocks = []
        for i, chunk in enumerate(chunks):
            block = (
                f"--- EVIDENCE [{i+1}] (Clause: {chunk.clause_id}) ---\n"
                f"{chunk.text}\n"
                f"Metadata: Issuer={chunk.metadata.get('issuer')}, "
                f"Status={chunk.metadata.get('approval_status')}, "
                f"EffectiveFrom={chunk.metadata.get('effective_from')}, "
                f"Locations={chunk.metadata.get('locations')}"
            )
            context_blocks.append(block)
        return "\n\n".join(context_blocks)

    def generate(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        snapshot_at: str,
        asked_at: str,
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generates grounded answer given question and retrieved context."""
        context_text = self.format_context(chunks)
        user_prompt = (
            f"Repository Snapshot Date: {snapshot_at}\n"
            f"Question Asked At: {asked_at}\n"
            f"User Context: {json.dumps(user_context)}\n\n"
            f"Retrieved Evidence:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Provide your JSON answer:"
        )

        if self.mock_mode:
            # Deterministic response generator for unit tests
            cited = [c.clause_id for c in chunks[:2]]
            return {
                "answer": f"Based on retrieved evidence: {chunks[0].text if chunks else 'No evidence'}",
                "cited_clauses": cited,
                "reasoning": "Synthesized from top retrieved chunks in mock mode."
            }

        self._load_model()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_response = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Parse JSON
        return self._parse_json_response(raw_response, chunks)

    def _parse_json_response(self, text: str, fallback_chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        """Extracts JSON object from model generation or provides safe fallback."""
        # Try finding json markdown code fence
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # Try finding raw { ... }
        match_raw = re.search(r"(\{.*\})", text, re.DOTALL)
        if match_raw:
            try:
                return json.loads(match_raw.group(1))
            except Exception:
                pass

        # Fallback if model output was not valid JSON
        cited = [c.clause_id for c in fallback_chunks[:2]]
        return {
            "answer": text,
            "cited_clauses": cited,
            "reasoning": "Extracted from raw model completion."
        }
