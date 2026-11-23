from pathlib import Path
import shutil
from datetime import datetime

target = Path('clipping/providers/groq.py')
if not target.exists():
    raise SystemExit('Run this script from the project root; clipping/providers/groq.py was not found.')

backup = target.with_name(f'groq.py.bak-chunking-{datetime.now():%Y%m%d-%H%M%S}')
shutil.copy2(target, backup)

new_code = r'''# Groq transport with transcript chunking for strict TPM limits.
import json
import re
import time
from openai import OpenAI


class GroqProvider:
    name = "groq"

    # Keep the complete request comfortably below Groq's 8k-token on-demand TPM cap.
    MAX_REQUEST_CHARS = 22000
    REQUEST_GAP_SECONDS = 65
    CLIP_TIME_OVERLAP_SECONDS = 179

    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def _split_transcript(transcript: str, max_chars: int) -> list[str]:
        lines = transcript.splitlines()
        if not lines:
            return [transcript]

        # Whisper transcript lines look like: [12.3 - 15.7] spoken words
        timed = []
        for line in lines:
            match = re.match(r"^\s*\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]", line)
            timed.append((line, float(match.group(1)) if match else None,
                          float(match.group(2)) if match else None))

        chunks = []
        start = 0
        while start < len(lines):
            end = start
            size = 0
            while end < len(lines) and (size + len(lines[end]) + 1 <= max_chars or end == start):
                size += len(lines[end]) + 1
                end += 1

            chunks.append("\n".join(lines[start:end]))
            if end >= len(lines):
                break

            # Keep up to 179 seconds of overlap so clips crossing a chunk
            # boundary can still be considered in full.
            last_end = timed[end - 1][2]
            next_start = end
            if last_end is not None:
                overlap_start = last_end - GroqProvider.CLIP_TIME_OVERLAP_SECONDS
                while next_start > start:
                    prev_start = timed[next_start - 1][1]
                    if prev_start is None or prev_start < overlap_start:
                        break
                    next_start -= 1
            start = max(start + 1, next_start)

        return chunks

    @staticmethod
    def _parse_result(content: str) -> list[dict]:
        if not content:
            raise ValueError("Groq returned an empty response.")
        result = json.loads(content)
        if isinstance(result, dict):
            for key in ("clips", "data", "highlights"):
                if isinstance(result.get(key), list):
                    result = result[key]
                    break
        if isinstance(result, dict):
            return [result]
        if not isinstance(result, list):
            raise ValueError(f"Groq returned unsupported JSON: {type(result).__name__}")
        return result

    def generate_clips(self, prompt: str, schema: dict) -> list[dict]:
        cfg = self.cfg
        if not cfg.api_key_groq:
            raise ValueError("GROQ_API_KEY not found in environment.")

        print(f"[3/3] Analyzing moments using Groq ({cfg.groq_model})...")
        schema_text = json.dumps(schema, ensure_ascii=False)
        if schema.get("type") == "array":
            instruction = 'Return a JSON object with one key "clips"; its value must be an array matching the supplied schema.'
        else:
            instruction = "Return one JSON object matching the supplied schema."

        system_content = (
            "You are a professional video editor and strategist. "
            "Return valid JSON only, no Markdown or commentary. "
            + instruction + "\nSchema:\n" + schema_text
        )
        marker = "Transkrip:\n"
        if marker not in prompt:
            raise ValueError("Could not find the transcript section in the analysis prompt.")
        prompt_prefix, transcript = prompt.rsplit(marker, 1)

        # Budget includes system instructions, schema, fixed prompt, and transcript.
        transcript_budget = self.MAX_REQUEST_CHARS - len(system_content) - len(prompt_prefix) - len(marker)
        if transcript_budget < 1500:
            raise ValueError(
                "Groq's fixed prompt/schema is too large for safe chunking "
                f"({len(system_content) + len(prompt_prefix)} characters)."
            )

        chunks = self._split_transcript(transcript, transcript_budget)
        print(f"Groq TPM protection: transcript split into {len(chunks)} sections.")

        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=cfg.api_key_groq,
            timeout=300.0,
            max_retries=2,
        )

        all_clips = []
        for index, chunk in enumerate(chunks, start=1):
            if index > 1:
                print(f"Waiting {self.REQUEST_GAP_SECONDS}s for Groq TPM window...")
                time.sleep(self.REQUEST_GAP_SECONDS)

            print(f"Groq analysis section {index}/{len(chunks)}...")
            chunk_prompt = prompt_prefix + marker + chunk
            # Ask for two candidates per section; the best three are selected globally.
            chunk_prompt = re.sub(r"Carikan\s+\d+\s+momen", "Carikan 2 momen", chunk_prompt, count=1)
            completion = client.chat.completions.create(
                model=cfg.groq_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": chunk_prompt},
                ],
                temperature=0.2,
                max_completion_tokens=4096,
                response_format={"type": "json_object"},
            )
            all_clips.extend(self._parse_result(completion.choices[0].message.content))

        # Merge candidates, remove duplicate timings, and keep the globally
        # highest-scoring requested clips.
        unique = {}
        for clip in all_clips:
            if not isinstance(clip, dict):
                continue
            key = (round(float(clip.get("start_time", -1)), 2),
                   round(float(clip.get("end_time", -1)), 2))
            if key not in unique or float(clip.get("viral_score", 0)) > float(unique[key].get("viral_score", 0)):
                unique[key] = clip

        ranked = sorted(
            unique.values(),
            key=lambda item: float(item.get("viral_score", 0)),
            reverse=True,
        )[: int(cfg.jumlah_clip)]
        for rank, clip in enumerate(ranked, start=1):
            clip["rank"] = rank
        return ranked
'''

target.write_text(new_code, encoding='utf-8')
print(f'Updated: {target}')
print(f'Backup:  {backup}')
