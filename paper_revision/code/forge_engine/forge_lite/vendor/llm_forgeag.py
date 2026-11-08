from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from protocol import FeatureSpec, compile_pointwise_expression

DEFAULT_MODEL = os.environ.get("FORGE_LLM_MODEL", "gpt-5.5")
DEFAULT_BACKEND = os.environ.get("FORGE_LLM_BACKEND", "codex").strip().lower()

POINTWISE_SCHEMA = {
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

AGG_SCHEMA = {
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
                    "op": {"type": "string"},
                    "entity_cols": {"type": "array", "items": {"type": "string"}},
                    "value_col": {"type": "string"},
                    "agg": {"type": "string"},
                    "window": {"type": "integer"},
                    "shift": {"type": "integer"},
                    "unit": {"type": "string"},
                    "encoding": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": [
                    "name", "op", "entity_cols", "value_col", "agg",
                    "window", "shift", "unit", "encoding", "description",
                ],
            },
        }
    },
    "required": ["features"],
}

CONSUMER_CONTEXT = (
    "You propose executable factor hypotheses for ONE Layer-1 member in the "
    "FORGE V13.1 loop. The evidence below is specific to that member's OOF "
    "binary-cross-entropy gradient residual. Your proposal has no authority "
    "to enter the model: it must pass a leakage/grammar firewall and compete "
    "with mechanical-bank candidates under the same cross-fitted FWL score. "
    "After tentative L1 surface updates, the complete 4-L1 -> 4-L2 -> "
    "weighted-ensemble graph is refit; the atomic round is accepted only if "
    "its held-out AUPRC delta reaches the frozen threshold. Bootstrap LCB is "
    "audit-only. Tree ensembles already learn thresholds and simple "
    "interactions natively, so target signal the CURRENT member measurably "
    "misses: cross-row structure, consistency/mismatch semantics, "
    "normalizations, and segment-specific patterns — not marginal monotone "
    "tweaks."
)


class ForgeAgLLMError(RuntimeError):
    pass


class ForgeAgLLM:
    def __init__(
        self,
        log_path: Path,
        *,
        model: str = DEFAULT_MODEL,
        timeout: int = 600,
        attempts: int = 2,
        backoff_sec: float = 5.0,
        max_calls: int = 40,
        stub: bool = False,
        backend: str | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.log_path = Path(log_path)
        self.model = model
        self.timeout = int(timeout)
        self.attempts = max(1, int(attempts))
        self.backoff_sec = float(backoff_sec)
        self.max_calls = int(max_calls)
        self.stub = bool(stub) or os.environ.get("FORGEAG_STUB_LLM") == "1"
        self.backend = (backend or DEFAULT_BACKEND).strip().lower()
        if self.backend in {"http", "openai", "openai_compatible"}:
            self.backend = "openai-compatible"
        if self.backend not in {"codex", "openai-compatible"}:
            raise ValueError(
                "FORGE_LLM_BACKEND must be 'codex' or 'openai-compatible'"
            )
        direct_url = api_url or os.environ.get("FORGE_LLM_API_URL", "").strip()
        base_url = os.environ.get("FORGE_LLM_BASE_URL", "").strip()
        self.api_url = direct_url or (
            f"{base_url.rstrip('/')}/chat/completions" if base_url else ""
        )
        self.api_key = (
            api_key if api_key is not None
            else os.environ.get("FORGE_LLM_API_KEY", "")
        )
        env_max_tokens = os.environ.get("FORGE_LLM_MAX_TOKENS", "4096")
        self.max_tokens = int(
            max_tokens if max_tokens is not None else env_max_tokens
        )
        self.user_agent = os.environ.get(
            "FORGE_LLM_USER_AGENT", "forge-cof/1.0"
        )
        if (not self.stub and self.backend == "openai-compatible"
                and not self.api_url):
            raise ValueError(
                "OpenAI-compatible backend requires FORGE_LLM_BASE_URL "
                "or FORGE_LLM_API_URL"
            )
        self.calls = 0
        self.requests = 0
        self.errors = 0
        self.tokens_in_est = 0
        self.tokens_out_est = 0
        self.scratch_cwd = tempfile.mkdtemp(prefix="forgeag_codex_scratch_")
        self._lock = threading.Lock()
        self.cli_version = "not-applicable"
        if self.stub:
            self.cli_version = "stub"
        elif self.backend == "codex":
            try:
                self.cli_version = subprocess.run(
                    ["codex", "-V"], capture_output=True, text=True, timeout=30,
                ).stdout.strip()
            except Exception:
                self.cli_version = "unknown"

    def _invoke_codex(self, prompt: str, schema: dict) -> tuple[str, str | None]:
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
                    cwd=self.scratch_cwd,
                    timeout=self.timeout,
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                )
                if output_path.exists():
                    raw = output_path.read_text(encoding="utf-8").strip()
                if completed.returncode != 0:
                    error = (f"codex return code {completed.returncode}: "
                             f"{completed.stderr[-1000:]}")
                elif not raw:
                    error = "empty response"
            except Exception as exc:
                error = repr(exc)
        return raw, error

    def _http_request(self, payload: dict) -> tuple[str, str | None, int | None]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib_request.Request(
            self.api_url, data=body, headers=headers, method="POST"
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8", errors="replace")
                status = getattr(response, "status", 200)
        except urllib_error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            return "", f"HTTP {exc.code}: {response_body[-1000:]}", exc.code
        except Exception as exc:
            return "", repr(exc), None

        try:
            envelope = json.loads(response_body)
            content = envelope["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict)
                )
            if not isinstance(content, str) or not content.strip():
                return "", "empty response content", status
            return content.strip(), None, status
        except Exception as exc:
            return "", f"invalid Chat Completions response: {exc!r}", status

    def _invoke_openai_compatible(
        self, prompt: str, schema: dict
    ) -> tuple[str, str | None]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "forge_features",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        raw, error, status = self._http_request(payload)
        if error is None:
            return raw, None

        if status not in {400, 404, 415, 422}:
            return raw, error
        payload["response_format"] = {"type": "json_object"}
        raw, fallback_error, _ = self._http_request(payload)
        if fallback_error is None:
            return raw, None
        return raw, (
            f"json_schema rejected ({error}); "
            f"fallback failed ({fallback_error})"
        )

    def _invoke_once(self, prompt: str, schema: dict) -> tuple[str, str | None]:
        if self.backend == "openai-compatible":
            return self._invoke_openai_compatible(prompt, schema)
        return self._invoke_codex(prompt, schema)

    def _log(self, *, kind: str, prompt: str, response: str,
             error: str | None, attempt: int, elapsed: float) -> None:
        tokens_in = len(prompt) // 4
        tokens_out = len(response) // 4
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.tokens_in_est += tokens_in
            self.tokens_out_est += tokens_out
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "kind": kind,
                    "model": "stub" if self.stub else self.model,
                    "backend": "stub" if self.stub else self.backend,
                    "cwd": self.scratch_cwd,
                    "attempt": attempt,
                    "elapsed_sec": round(elapsed, 3),
                    "prompt": prompt,
                    "response": response,
                    "tokens_in_est": tokens_in,
                    "tokens_out_est": tokens_out,
                    "error": error,
                }, ensure_ascii=False) + "\n")

    def call_json(self, kind: str, prompt: str, schema: dict) -> dict:
        with self._lock:
            self.requests += 1
        for attempt in range(1, self.attempts + 1):
            with self._lock:
                over_cap = self.calls >= self.max_calls
                if not over_cap:
                    self.calls += 1
            if over_cap:
                self._log(kind=kind, prompt=prompt, response="",
                          error=f"llm call cap {self.max_calls} reached",
                          attempt=attempt, elapsed=0.0)
                return {}
            started = time.time()
            raw, error = self._invoke_once(prompt, schema)
            parsed: dict = {}
            if raw and error is None:
                try:
                    start, end = raw.find("{"), raw.rfind("}")
                    parsed = (json.loads(raw[start:end + 1])
                              if start >= 0 and end >= start else {})
                except Exception as exc:
                    error = f"invalid JSON: {exc!r}"
            self._log(kind=kind, prompt=prompt, response=raw, error=error,
                      attempt=attempt, elapsed=time.time() - started)
            if error is None and parsed:
                return parsed
            with self._lock:
                self.errors += 1
            if attempt < self.attempts:
                time.sleep(self.backoff_sec * attempt)
        return {}

    def _stub_pointwise(self, columns: list[str], tried: set[str],
                        n: int) -> list[FeatureSpec]:
        specs: list[FeatureSpec] = []
        usable = [c for c in columns if c][:12]
        index = 0
        while len(specs) < n and index < 40:
            left = usable[index % len(usable)]
            right = usable[(index + 1) % len(usable)]
            name = f"stub_pw_{index:02d}_{left[:12]}"
            expression = (f"df['{left}'] / (df['{right}'].abs() + 1.0)"
                          if index % 2 == 0 else
                          f"np.log1p(df['{left}'].abs())")
            index += 1
            if name in tried:
                continue
            specs.append(FeatureSpec(
                name=name, expression=expression, family="stub",
                source="stub_pointwise", description="stub candidate",
            ))
        return specs

    def _stub_aggregates(self, entity_candidates: list[str], tried: set[str],
                         n: int) -> list[dict]:
        out: list[dict] = []
        for index, entity in enumerate(entity_candidates[: n * 2]):
            if len(out) >= n:
                break
            for op in ("cum_count", "time_since_prev"):
                name = f"stub_agg_{op}_{entity[:12]}"
                if name in tried or any(o["name"] == name for o in out):
                    continue
                out.append({
                    "name": name, "op": op, "entity_cols": [entity],
                    "value_col": "", "agg": "", "window": 1, "shift": 1,
                    "unit": "", "encoding": "",
                    "description": "stub aggregation candidate",
                    "source": "stub_agg",
                })
                break
        return out[:n]

    def propose_pointwise(
        self,
        *,
        dataset_description: str,
        label: str,
        columns: list[str],
        existing_features: list[str],
        tried: set[str],
        residual_profile: str,
        n: int,
        source: str = "ag_residual_pointwise",
    ) -> list[FeatureSpec]:
        if self.stub:
            return self._stub_pointwise(existing_features, tried, n)
        visible = [c for c in columns if c != label][:120]
        prompt = f"""{CONSUMER_CONTEXT}

Dataset: {dataset_description}
Target column (FORBIDDEN in expressions): {label}
Raw and already-created columns available: {', '.join(visible)}
Current model features: {', '.join(existing_features[:100])}
Already tried names (do not repeat): {', '.join(sorted(tried)[:120])}

{residual_profile}

Return {n} distinct numeric feature expressions. Each expression must:
- use only df['column'], np.*, pd.to_numeric, and vectorized Series methods;
- be pointwise: no groupby, transform, rank, rolling, expanding, merge, or join;
- never reference the target column;
- guard division and logarithms, allow NaN, and never use row-wise apply/lambda;
- use a new snake_case name and include a concise reason.

Output only the requested JSON object."""
        parsed = self.call_json("pointwise", prompt, POINTWISE_SCHEMA)
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
                name=name, expression=expression,
                family=str(item.get("family", "llm")), source=source,
                description=str(item.get("description", "")),
            ))
        return proposals[:n]

    def propose_aggregates(
        self,
        *,
        dataset_description: str,
        label: str,
        entity_candidates: list[str],
        value_candidates: list[str],
        existing_features: list[str],
        tried: set[str],
        residual_profile: str,
        n: int,
        source: str = "ag_residual_agg",
    ) -> list[dict]:
        if self.stub:
            return self._stub_aggregates(entity_candidates, tried, n)
        ops_help = (
            "- lag: per-entity previous value of value_col (shift k >= 1)\n"
            "- rolling_stat: per-entity backward window of `window` rows of "
            "value_col after shift >= 1; agg in {mean,sum,std,max,min}\n"
            "- expanding_stat: per-entity cumulative stat of value_col up to "
            "the previous row; agg in {mean,sum,std,max,min}\n"
            "- cum_count: per-entity running count of strictly earlier events; "
            "no value_col needed\n"
            "- time_since_prev: time units since the entity's previous event; "
            "no value_col needed\n"
            "- group_ratio: value_col divided by the entity's expanding mean "
            "of strictly earlier values\n"
            "- periodic: cyclic encoding of the row's own timestamp; set "
            "unit in {hour,dayofweek} and encoding in {sin,cos,raw}; "
            "entity_cols/value_col are ignored"
        )
        prompt = f"""{CONSUMER_CONTEXT}

You are proposing leakage-safe ENTITY AGGREGATION features (velocity, reuse,
consistency per entity) for the same task. Rows are processed in strict
chronological order; every op is causal (uses only strictly earlier rows of
the same entity).

Dataset: {dataset_description}
Target column (FORBIDDEN everywhere): {label}
Entity column candidates (choose 1-2 per feature): {', '.join(entity_candidates[:24])}
Numeric value column candidates: {', '.join(value_candidates[:24])}
Current model features: {', '.join(existing_features[:80])}
Already tried names (do not repeat): {', '.join(sorted(tried)[:120])}

{residual_profile}

Allowed ops:
{ops_help}

Return up to {n} structured aggregation specs. For each spec:
- name: new snake_case name;
- op: one of the allowed ops above;
- entity_cols: 1-2 entity columns from the candidates ([] for periodic);
- value_col: a numeric value column (use "" for ops that need none);
- agg: one of mean/sum/std/max/min (use "" when the op takes no agg);
- window: integer 1-10000 (use 1 when the op takes no window);
- shift: integer 1-10000 (usually 1);
- unit / encoding: only for periodic (else "");
- description: one concise reason.
Never reference the target column. Output only the requested JSON object."""
        parsed = self.call_json("aggregate", prompt, AGG_SCHEMA)
        proposals: list[dict] = []
        for item in parsed.get("features", []):
            name = str(item.get("name", "")).strip().lower().replace(" ", "_")
            if not name or name in tried:
                continue
            entity_cols = item.get("entity_cols", [])
            if isinstance(entity_cols, str):
                entity_cols = [entity_cols]
            proposals.append({
                "name": name,
                "op": str(item.get("op", "")).strip().lower(),
                "entity_cols": [str(c).strip().lower() for c in entity_cols],
                "value_col": str(item.get("value_col", "") or "").strip().lower(),
                "agg": str(item.get("agg", "") or "").strip().lower(),
                "window": item.get("window"),
                "shift": item.get("shift"),
                "unit": str(item.get("unit", "") or "").strip().lower(),
                "encoding": str(item.get("encoding", "") or "").strip().lower(),
                "description": str(item.get("description", "")),
                "source": source,
            })
        return proposals[:n]

    def audit(self) -> dict:
        return {
            "backend": "codex-cli-structured",
            "model": "stub" if self.stub else self.model,
            "codex_cli_version": self.cli_version,
            "stub": self.stub,
            "scratch_cwd": self.scratch_cwd,
            "sandbox": "read-only",
            "ephemeral": True,
            "requests": self.requests,
            "calls": self.calls,
            "max_calls": self.max_calls,
            "errors": self.errors,
            "tokens_in_est": self.tokens_in_est,
            "tokens_out_est": self.tokens_out_est,
            "log_path": str(self.log_path),
        }
