"""
VAL API service: wraps the KCL-Planning VAL 'validate' binary over HTTP.

POST /validate
  Body: { "domain": "<pddl str>", "problem": "<pddl str>", "plan": "<plan str>", "verbose": false }
  Response: { "valid": bool, "output": "<raw validate stdout>" }

GET /health
  Response: { "status": "ok", "val_path": "<path>" }
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="VAL API", version="1.0.0")

VAL_PATH = os.environ.get("VAL", "/usr/local")
_VALIDATE_BIN = os.path.join(VAL_PATH, "bin", "validate")


def _find_validate() -> Optional[str]:
    for candidate in [
        _VALIDATE_BIN,
        os.path.join(VAL_PATH, "validate"),
        "/usr/local/bin/validate",
        "/usr/bin/validate",
    ]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


class ValidateRequest(BaseModel):
    domain: str
    problem: str
    plan: str
    verbose: bool = False


class ValidateResponse(BaseModel):
    valid: bool
    output: str


@app.get("/health")
def health():
    binary = _find_validate()
    return {"status": "ok" if binary else "degraded", "val_path": VAL_PATH, "validate_binary": binary}


@app.post("/validate", response_model=ValidateResponse)
def validate(req: ValidateRequest) -> ValidateResponse:
    binary = _find_validate()
    if not binary:
        raise HTTPException(
            status_code=503,
            detail=f"VAL 'validate' binary not found. Set VAL env to the directory containing it. Searched: {VAL_PATH}",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        domain_file = os.path.join(tmpdir, "domain.pddl")
        problem_file = os.path.join(tmpdir, "problem.pddl")
        plan_file = os.path.join(tmpdir, "plan.txt")

        with open(domain_file, "w") as f:
            f.write(req.domain)
        with open(problem_file, "w") as f:
            f.write(req.problem)
        with open(plan_file, "w") as f:
            f.write(req.plan)

        cmd = [binary]
        if req.verbose:
            cmd.append("-v")
        cmd.extend([domain_file, problem_file, plan_file])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout + (result.stderr or "")
        valid = "Plan valid" in output

        return ValidateResponse(valid=valid, output=output)
