"""A stand-in for an LLM command-line tool (used by test_llm_modes.py).

Reads the prompt on stdin and prints a JSON answer on stdout: R0 features or
CoF templates, depending on the schema stated at the end of the prompt.
"""

import json
import sys

prompt = sys.stdin.read()
if '"templates"' in prompt.rsplit("JSON schema:", 1)[-1]:
    answer = {"templates": [
        {"name": "inter", "kind": "pointwise", "mechanism": "interaction", "form": "{A}*{B}",
         "slots": [{"slot": "A", "columns": ["x1", "*"]}, {"slot": "B", "columns": ["x2"]}],
         "entities": [], "values": [], "aggregations": [], "windows": [], "shift": 1},
        {"name": "hist", "kind": "aggregate", "mechanism": "customer history", "form": "", "slots": [],
         "entities": ["customer"], "values": ["debt"], "aggregations": ["mean", "std"], "windows": [0, 4],
         "shift": 1}]}
else:
    family = next(f for f in ("unary", "binary", "ternary", "related_column", "complement")
                  if f"{f} numeric" in prompt)
    assert family != "complement" or "already_exploited" in prompt
    answer = {"features": [{"name": f"{family}_burden", "family": family, "description": "burden",
                            "python_expr": "df['debt']/(np.abs(df['income'])+1.0)"}]}
print("Here is the answer:\n```json\n" + json.dumps(answer) + "\n```")
