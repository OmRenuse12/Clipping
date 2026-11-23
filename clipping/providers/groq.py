# Groq transport with transcript chunking for strict TPM limits.
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

        print(f"[3/3] Finding the single best clip using Groq ({cfg.groq_model})...")
        schema_text = json.dumps(schema, ensure_ascii=False)
        marker = "Transkrip:\n"
        if marker not in prompt:
            raise ValueError("Could not find transcript in analysis prompt.")

        _, transcript = prompt.rsplit(marker, 1)

        system_content = (
            "You are an expert viral short-form video editor. "
            "Return valid JSON only. Select the single strongest self-contained "
            "20-179 second moment from the supplied transcript. Prioritize a "
            "powerful hook, emotion, surprising insight, useful information, "
            "storytelling, and shareability. Use exact transcript timestamps. "
            "Do not invent dialogue or timestamps. Return one candidate only. "
            "Follow this JSON schema: " + schema_text
        )

        chunks = self._split_transcript(transcript, 14000)
        print(f"Transcript divided into {len(chunks)} sections.")

        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=cfg.api_key_groq,
            timeout=300.0,
            max_retries=2,
        )

        candidates = []
        for index, chunk in enumerate(chunks, start=1):
            if index > 1:
                print("Waiting 65 seconds for Groq rate limit...")
                time.sleep(65)

            print(f"Evaluating section {index}/{len(chunks)}...")
            completion = client.chat.completions.create(
                model=cfg.groq_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": "Find the best single clip in this section:\\n" + chunk},
                ],
                temperature=0.2,
                max_completion_tokens=1200,
                response_format={"type": "json_object"},
            )
            candidates.extend(self._parse_result(completion.choices[0].message.content))

        valid = []
        for clip in candidates:
            if not isinstance(clip, dict):
                continue
            try:
                start_time = float(clip.get("start_time", -1))
                end_time = float(clip.get("end_time", -1))
                score = float(clip.get("viral_score", 0))
                if 0 <= start_time < end_time and 20 <= end_time - start_time <= 179:
                    valid.append((score, clip))
            except (TypeError, ValueError):
                continue

        if not valid:
            raise ValueError("Groq did not return a valid clip. Check the analysis output.")

        best = max(valid, key=lambda item: item[0])[1]
        best["rank"] = 1
        print("Selected the highest-scoring clip from all sections.")
        return [best]

