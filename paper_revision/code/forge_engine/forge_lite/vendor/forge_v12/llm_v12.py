from __future__ import annotations

from llm_forgeag import (CONSUMER_CONTEXT, POINTWISE_SCHEMA, ForgeAgLLM,
                         FeatureSpec, compile_pointwise_expression)


class VariableBundleLLM(ForgeAgLLM):

    def propose_pointwise_variable(
        self, *, dataset_description: str, label: str, columns: list[str],
        existing_features: list[str], tried: set[str], residual_profile: str,
        max_n: int, source: str,
    ) -> list[FeatureSpec]:
        if max_n < 0:
            raise ValueError("max_n must be non-negative")
        if max_n == 0:
            return []
        if self.stub:
            return self._stub_pointwise(existing_features, tried, max_n)
        visible = [c for c in columns if c != label][:120]
        prompt = f"""{CONSUMER_CONTEXT}

Dataset: {dataset_description}
Target column (FORBIDDEN): {label}
Available columns: {', '.join(visible)}
Current consumer features: {', '.join(existing_features[:100])}
Already tried names: {', '.join(sorted(tried)[:120])}

{residual_profile}

Propose between 0 and {max_n} distinct numeric pointwise features. This is
not a top-k task: return an empty features array when no defensible residual
mechanism exists. Each expression must use only df['column'], np.*,
pd.to_numeric and vectorized Series methods; it must not use groupby, rank,
rolling, expanding, merge, join, apply/lambda, or the target. Guard division
and logarithms. Series.eq/gt/ge/lt/le followed by astype('float64') are
allowed for exact-value and bounded-segment indicators;
np.sin, np.cos and np.arctan2 are allowed for row-wise harmonic features.
Use new snake_case names and explain the missed mechanism.
Output only the requested JSON object."""
        parsed = self.call_json("v12_variable_pointwise", prompt,
                                POINTWISE_SCHEMA)
        proposals: list[FeatureSpec] = []
        for item in parsed.get("features", []):
            name = str(item.get("name", "")).strip().lower().replace(" ", "_")
            expression = str(item.get("python_expr", "")).strip()
            if not name or not expression or name in tried:
                continue
            try:
                compile_pointwise_expression(expression,
                                             forbidden_columns=(label,))
            except Exception:
                continue
            proposals.append(FeatureSpec(
                name=name, expression=expression,
                family=str(item.get("family", "llm")), source=source,
                description=str(item.get("description", ""))))
        return proposals[:max_n]
