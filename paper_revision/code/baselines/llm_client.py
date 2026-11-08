from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from protocol import FeatureSpec, compile_pointwise_expression


FEATURE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "features": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "family": {"type": "string"},
                    "description": {"type": "string"},
                    "python_expr": {"type": "string"},
                },
                "required": ["name", "family", "description", "python_expr"],
            },
        }
    },
    "required": ["features"],
}


class StructuredLLM:
    def __init__(self, log_path: Path, timeout: int = 600) -> None:
        self.log_path = Path(log_path)
        self.timeout = int(timeout)
        self.calls = 0
        self.tokens_in_est = 0
        self.tokens_out_est = 0
        self.model = os.environ.get("FORGE_LLM_MODEL", "gpt-5.5")
        self.errors = 0
        try:
            self.cli_version = subprocess.run(
                ["codex", "-V"], capture_output=True, text=True, timeout=30,
            ).stdout.strip()
        except Exception:
            self.cli_version = "unknown"

    def call_json(self, prompt: str, schema: dict = FEATURE_SCHEMA) -> tuple[dict, str]:
        self.calls += 1
        raw = ""
        error = None
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            schema_path = directory_path / "schema.json"
            output_path = directory_path / "output.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            command = [
                "codex", "exec", "--skip-git-repo-check",
                "--model", self.model,
                "--sandbox", "read-only",
                "--ephemeral",
                "--color", "never",
                "-c", "model_reasoning_effort=medium",
                "--output-schema", str(schema_path),
                "--output-last-message", str(output_path),
                prompt,
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(Path(__file__).resolve().parent),
                    timeout=self.timeout,
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                )
                if output_path.exists():
                    raw = output_path.read_text(encoding="utf-8").strip()
                if completed.returncode != 0:
                    error = f"codex return code {completed.returncode}: {completed.stderr[-1000:]}"
            except Exception as exc:
                error = repr(exc)

        parsed: dict = {}
        if raw:
            try:
                start, end = raw.find("{"), raw.rfind("}")
                parsed = json.loads(raw[start:end + 1]) if start >= 0 and end >= start else {}
            except Exception as exc:
                error = f"invalid JSON: {exc!r}"

        tokens_in = len(prompt) // 4
        tokens_out = len(raw) // 4
        self.tokens_in_est += tokens_in
        self.tokens_out_est += tokens_out
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "model": self.model,
                    "prompt": prompt,
                    "response": raw,
                    "tokens_in_est": tokens_in,
                    "tokens_out_est": tokens_out,
                    "error": error,
                },
                ensure_ascii=False,
            ) + "\n")
        if error is not None:
            self.errors += 1
            raise RuntimeError(f"structured LLM call failed: {error}")
        return parsed, raw

    def propose(
        self,
        *,
        dataset_description: str,
        label: str,
        columns: list[str],
        existing_features: list[str],
        tried: set[str],
        guidance: str,
        n: int,
        source: str,
    ) -> list[FeatureSpec]:
        visible_columns = [column for column in columns if column != label][:120]
        prompt = f"""You are implementing leakage-safe pointwise feature engineering for a
binary tabular classification task.

Dataset: {dataset_description}
Target column (FORBIDDEN in expressions): {label}
Raw and already-created columns available to this state: {', '.join(visible_columns)}
Current model features: {', '.join(existing_features[:100])}
Already tried names (do not repeat): {', '.join(sorted(tried)[:120])}

Guidance:
{guidance}

Return {n} distinct numeric feature expressions. Each expression must:
- use only df['column'], np.*, pd.to_numeric, and vectorized Series methods;
- be pointwise: no groupby, transform, rank, rolling, expanding, merge, or join;
- never reference the target column;
- guard division and logarithms, allow NaN, and never use row-wise apply/lambda;
- use a new snake_case name and include a concise reason.

Output only the requested JSON object."""
        parsed, _ = self.call_json(prompt)
        proposals: list[FeatureSpec] = []
        for item in parsed.get("features", []):
            name = str(item.get("name", "")).strip().lower().replace(" ", "_")
            expression = str(item.get("python_expr", "")).strip()
            if not name or not expression or name in tried:
                continue
            try:
                compile_pointwise_expression(expression, forbidden_columns=(label,))
            except Exception:
                continue
            proposals.append(FeatureSpec(
                name=name,
                expression=expression,
                family=str(item.get("family", "llm")),
                source=source,
                description=str(item.get("description", "")),
            ))
        return proposals[:n]

    def audit(self) -> dict:
        return {
            "model": self.model,
            "codex_cli_version": self.cli_version,
            "sandbox": "read-only",
            "ephemeral": True,
            "calls": self.calls,
            "tokens_in_est": self.tokens_in_est,
            "tokens_out_est": self.tokens_out_est,
            "errors": self.errors,
            "log_path": str(self.log_path),
        }
