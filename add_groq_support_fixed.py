from pathlib import Path
from datetime import datetime
import shutil

ROOT = Path.cwd()
config = ROOT / "clipping" / "config.py"
engine = ROOT / "clipping" / "engine.py"
validation = ROOT / "clipping" / "providers" / "validation.py"
groq_file = ROOT / "clipping" / "providers" / "groq.py"

for path in (config, engine, validation):
    if not path.is_file():
        raise SystemExit(f"Run from project root; missing {path}")

def prepare(path, replacements):
    original = path.read_text(encoding="utf-8")
    updated = original
    for old, new, label in replacements:
        count = updated.count(old)
        if count != 1:
            raise SystemExit(f"STOP: {label}: expected one match, found {count}. No files changed.")
        updated = updated.replace(old, new, 1)
    return original, updated

config_replacements = [
    ('AI_PROVIDER = "gemini"\nNVIDIA_MODEL = "deepseek-ai/deepseek-v4-pro"\nGEMINI_MODEL = "gemini-3-flash-preview"',
     'AI_PROVIDER = "gemini"\nNVIDIA_MODEL = "deepseek-ai/deepseek-v4-pro"\nGROQ_MODEL = "openai/gpt-oss-120b"\nGEMINI_MODEL = "gemini-3-flash-preview"', "Groq model default"),
    ('choices=["gemini", "nvidia", "vllm"],', 'choices=["gemini", "nvidia", "vllm", "groq"],', "provider choices"),
    ('    p.add_argument("--gemini-model", default=GEMINI_MODEL, help="Gemini model name")',
     '    p.add_argument(\n        "--groq-model",\n        default=GROQ_MODEL,\n        help="Groq model name (default: openai/gpt-oss-120b)",\n    )\n    p.add_argument("--gemini-model", default=GEMINI_MODEL, help="Gemini model name")', "Groq CLI model option"),
    ('        api_key_nvidia=os.environ.get("NVIDIA_API_KEY", ""),\n        nvidia_model=args.nvidia_model,',
     '        api_key_nvidia=os.environ.get("NVIDIA_API_KEY", ""),\n        api_key_groq=os.environ.get("GROQ_API_KEY", ""),\n        groq_model=args.groq_model,\n        nvidia_model=args.nvidia_model,', "Groq config fields"),
]

engine_replacements = [
    ('def analyze_with_vllm(transkrip_lengkap: str, cfg) -> list[dict]:',
     'def analyze_with_groq(transkrip_lengkap: str, cfg) -> list[dict]:\n'
     '    """Analyze transcript using Groq."""\n'
     '    from clipping.providers.groq import GroqProvider\n'
     '    from clipping.providers.schema import build_clips_schema\n\n'
     '    prompt = get_analysis_prompt(\n'
     '        transkrip_lengkap, cfg.jumlah_clip, cfg.durasi_hook, cfg=cfg\n'
     '    )\n'
     '    return GroqProvider(cfg).generate_clips(\n'
     '        prompt, build_clips_schema(TARGET_ACCOUNTS)\n'
     '    )\n\n\n'
     'def analyze_with_vllm(transkrip_lengkap: str, cfg) -> list[dict]:', "Groq analysis function"),
    ('    if provider == "vllm":\n        return analyze_with_vllm(transkrip_lengkap, cfg)\n\n    if provider == "nvidia":',
     '    if provider == "vllm":\n        return analyze_with_vllm(transkrip_lengkap, cfg)\n\n'
     '    if provider == "groq":\n        return analyze_with_groq(transkrip_lengkap, cfg)\n\n'
     '    if provider == "nvidia":', "Groq engine dispatch"),
]

validation_replacements = [
    ('    if provider not in {"gemini", "nvidia"}:', '    if provider not in {"gemini", "nvidia", "groq"}:', "provider validation allow-list"),
    ('    if not getattr(cfg, "api_key_gemini", ""):\n        key = "NVIDIA_API_KEY or GOOGLE_API_KEY" if provider == "nvidia" else "GOOGLE_API_KEY"\n        raise ProviderConfigError(f"{key} tidak ditemukan. Set via Settings atau environment.")',
     '    if provider == "groq":\n'
     '        if not getattr(cfg, "api_key_groq", ""):\n'
     '            raise ProviderConfigError("GROQ_API_KEY tidak ditemukan. Tambahkan ke .env atau environment.")\n'
     '        return\n'
     '    if not getattr(cfg, "api_key_gemini", ""):\n'
     '        key = "NVIDIA_API_KEY or GOOGLE_API_KEY" if provider == "nvidia" else "GOOGLE_API_KEY"\n'
     '        raise ProviderConfigError(f"{key} tidak ditemukan. Set via Settings atau environment.")', "provider API key validation"),
]

changes = {}
for path, replacements in ((config, config_replacements), (engine, engine_replacements), (validation, validation_replacements)):
    changes[path] = prepare(path, replacements)

groq_code = """\
# Groq transport using the project existing OpenAI-compatible SDK.
import json
from openai import OpenAI

class GroqProvider:
    name = "groq"

    def __init__(self, cfg):
        self.cfg = cfg

    def generate_clips(self, prompt: str, schema: dict) -> list[dict]:
        cfg = self.cfg
        if not cfg.api_key_groq:
            raise ValueError("GROQ_API_KEY not found in environment.")
        print(f"[3/3] Analyzing Top {cfg.jumlah_clip} moments using Groq ({cfg.groq_model})...")
        schema_text = json.dumps(schema, ensure_ascii=False)
        if schema.get("type") == "array":
            instruction = 'Return a JSON object with one key "clips"; its value must be an array matching the supplied schema.'
        else:
            instruction = "Return one JSON object matching the supplied schema."
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=cfg.api_key_groq, timeout=300.0, max_retries=2)
        completion = client.chat.completions.create(
            model=cfg.groq_model,
            messages=[
                {"role": "system", "content": "You are a professional video editor and strategist. Return valid JSON only, no Markdown or commentary. " + instruction + "\\nSchema:\\n" + schema_text},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=16384,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
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
"""

if groq_file.exists():
    existing = groq_file.read_text(encoding="utf-8")
    changes[groq_file] = (existing, groq_code)
else:
    changes[groq_file] = ("", groq_code)

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
for path, (original, updated) in changes.items():
    if original == updated:
        continue
    if original:
        backup = path.with_name(path.name + f".bak-{stamp}")
        shutil.copy2(path, backup)
        print(f"Backup created: {backup.name}")
    path.write_text(updated, encoding="utf-8")
    print(f"Updated: {path.relative_to(ROOT)}")

print("\nGroq integration added.")
print("Add GROQ_API_KEY=your_key to .env, then run: python main.py --help")
print('Confirm "groq" appears in the --ai-provider choices.')
