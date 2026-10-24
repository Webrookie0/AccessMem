"""Run a role-aware evaluation and optionally score answers with RAGAS."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from contextlib import asynccontextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.auth.rbac import can_user_read
from app.auth.users import get_user
from app.config import get_settings
from app.services.guardrails import check_input
from app.services.rag import answer_question
from app.services.vector_store import secure_search

DEFAULT_DATASET = Path(__file__).with_name("dataset.jsonl")


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    username: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=4000)
    reference: str = Field(min_length=1)
    expected_sources: list[str] = Field(default_factory=list)
    expected_refusal: bool = False
    expected_blocked: bool = False

    @model_validator(mode="after")
    def check_expectations(self):
        if (self.expected_refusal or self.expected_blocked) and self.expected_sources:
            raise ValueError("Refusal and blocked cases must not expect answer sources")
        if not self.expected_refusal and not self.expected_blocked and not self.expected_sources:
            raise ValueError("Answerable cases require expected sources")
        return self


def load_dataset(path: Path) -> list[EvaluationCase]:
    cases = [
        EvaluationCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Dataset must be non-empty and case IDs must be unique")
    return cases


@asynccontextmanager
async def ragas_metrics():
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError("Install requirements-eval.txt to use RAGAS") from exc
    settings = get_settings()
    async with AsyncOpenAI(
        base_url=settings.ragas_base_url,
        api_key=settings.ragas_api_key,
        timeout=settings.ragas_timeout,
        max_retries=0,
    ) as client:
        llm = llm_factory(settings.ragas_model, client=client, temperature=0)
        embeddings = OpenAIEmbeddings(client=client, model=settings.ragas_embed_model)
        yield {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
        }


async def _score_answer(
    metrics: dict, case: EvaluationCase, answer: str, contexts: list[str]
) -> tuple[dict, dict]:
    values, errors = {}, {}
    arguments = {
        "faithfulness": {
            "user_input": case.question,
            "response": answer,
            "retrieved_contexts": contexts,
        },
        "answer_relevancy": {"user_input": case.question, "response": answer},
        "context_precision": {
            "user_input": case.question,
            "reference": case.reference,
            "retrieved_contexts": contexts,
        },
        "context_recall": {
            "user_input": case.question,
            "reference": case.reference,
            "retrieved_contexts": contexts,
        },
    }
    for name, metric in metrics.items():
        try:
            value = float((await metric.ascore(**arguments[name])).value)
            if not math.isfinite(value):
                raise ValueError("Non-finite metric result")
            values[name] = value
        except Exception as exc:
            # Evaluation must retain failed metrics without leaking provider payloads.
            errors[name] = type(exc).__name__
    return values, errors


def _summarize(records: list[dict], *, retrieval_only: bool) -> dict:
    summary = {
        "cases": len(records),
        "run_errors": sum("error" in record for record in records),
        "unauthorized_chunks": sum(record.get("unauthorized_chunks", 0) for record in records),
        "ragas_metric_errors": sum(len(record.get("metric_errors", {})) for record in records),
        "input_guardrail_failures": sum(
            record.get("input_guardrail_failed", False) for record in records
        ),
    }
    for field in ("source_recall_at_k", "source_mrr"):
        values = [record[field] for record in records if record.get(field) is not None]
        if values:
            summary[field] = round(mean(values), 4)
            summary[field + "_samples"] = len(values)
    latencies = [record["latency_ms"] for record in records if "error" not in record]
    if latencies:
        summary["median_latency_ms"] = round(median(latencies), 2)
    if not retrieval_only:
        valid = [record for record in records if "error" not in record]
        summary["behavior_passes"] = sum(record["behavior_correct"] for record in valid)
        summary["behavior_samples"] = len(valid)
        refused = [record for record in valid if record["expected_refusal"]]
        if refused:
            summary["correct_refusals"] = sum(
                record["refused"] and not record["blocked"] for record in refused
            )
            summary["expected_refusals"] = len(refused)
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        values = [
            record["metrics"][name] for record in records if name in record.get("metrics", {})
        ]
        if values:
            summary[name] = round(mean(values), 4)
            summary[name + "_samples"] = len(values)
    return summary


async def _run_evaluation(
    cases: list[EvaluationCase],
    *,
    modes: list[str],
    retrieval_only: bool = False,
    with_ragas: bool = False,
    include_text: bool = False,
) -> dict:
    if with_ragas and retrieval_only:
        raise ValueError("RAGAS answer metrics require full question answering")
    if not modes or any(mode not in {"dense", "hybrid"} for mode in modes):
        raise ValueError("Evaluation modes must be dense or hybrid")
    records = []
    context = ragas_metrics() if with_ragas else nullcontext({})
    async with context as metrics:
        for mode in modes:
            for case in cases:
                user = get_user(case.username)
                if user is None:
                    raise ValueError(f"Evaluation account is unavailable: {case.username}")
                started = perf_counter()
                record = {
                    "case_id": case.id,
                    "username": user.username,
                    "role": user.role,
                    "tenant_id": user.tenant_id,
                    "mode": mode,
                    "question_hash": hashlib.sha256(case.question.encode()).hexdigest(),
                    "expected_refusal": case.expected_refusal,
                    "expected_blocked": case.expected_blocked,
                    "expected_sources": case.expected_sources,
                }
                try:
                    if retrieval_only:
                        blocked = not check_input(case.question).allowed
                        chunks = [] if blocked else secure_search(user, case.question, mode=mode)
                        answer, refused = "", False
                    else:
                        result = answer_question(user, case.question, mode=mode)
                        chunks, answer = result.retrieved_chunks, result.answer
                        blocked, refused = result.blocked, result.refused
                    sources = list(dict.fromkeys(chunk.metadata["source"] for chunk in chunks))
                    record.update(
                        sources=sources,
                        blocked=blocked,
                        refused=refused,
                        unauthorized_chunks=sum(
                            not can_user_read(user, chunk.metadata) for chunk in chunks
                        ),
                    )
                    if record["unauthorized_chunks"]:
                        raise PermissionError("Unauthorized context must not be evaluated")
                    record["input_guardrail_failed"] = case.expected_blocked and not blocked
                    if case.expected_sources:
                        found = set(sources).intersection(case.expected_sources)
                        record["source_recall_at_k"] = len(found) / len(set(case.expected_sources))
                        record["source_mrr"] = next(
                            (1 / rank for rank, source in enumerate(sources, 1) if source in found),
                            0.0,
                        )
                    if not retrieval_only:
                        record["behavior_correct"] = (
                            blocked
                            if case.expected_blocked
                            else refused and not blocked
                            if case.expected_refusal
                            else not refused and not blocked
                        )
                    contexts = [chunk.text for chunk in chunks]
                    if with_ragas and not (
                        blocked or refused or case.expected_refusal or case.expected_blocked
                    ):
                        values, errors = await _score_answer(metrics, case, answer, contexts)
                        record.update(metrics=values, metric_errors=errors)
                    if include_text:
                        record.update(
                            question=case.question, answer=answer, retrieved_contexts=contexts
                        )
                except Exception as exc:
                    record["error"] = type(exc).__name__
                record["latency_ms"] = round((perf_counter() - started) * 1000, 2)
                records.append(record)
    groups = {
        f"{mode}/{tenant}/{role}": _summarize(
            [
                record
                for record in records
                if record["mode"] == mode
                and record["tenant_id"] == tenant
                and record["role"] == role
            ],
            retrieval_only=retrieval_only,
        )
        for mode, tenant, role in sorted(
            {(record["mode"], record["tenant_id"], record["role"]) for record in records}
        )
    }
    return {
        "project": "AccessMem",
        "created_at": datetime.now(UTC).isoformat(),
        "retrieval_only": retrieval_only,
        "ragas_enabled": with_ragas,
        "top_k": get_settings().top_k,
        "summaries": {
            mode: _summarize(
                [record for record in records if record["mode"] == mode],
                retrieval_only=retrieval_only,
            )
            for mode in modes
        },
        "groups": groups,
        "records": records,
    }


def run_evaluation(
    cases: list[EvaluationCase],
    *,
    modes: list[str],
    retrieval_only: bool = False,
    with_ragas: bool = False,
    include_text: bool = False,
) -> dict:
    return asyncio.run(
        _run_evaluation(
            cases,
            modes=modes,
            retrieval_only=retrieval_only,
            with_ragas=with_ragas,
            include_text=include_text,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AccessMem by user, tenant and retrieval mode"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mode", choices=["dense", "hybrid", "both"], default="both")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--ragas", action="store_true")
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include question and retrieved text in the local report",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("eval/results/report.json"))
    args = parser.parse_args()
    if args.ragas and args.retrieval_only:
        parser.error("--ragas cannot be combined with --retrieval-only")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        cases = load_dataset(args.dataset)
        if args.limit:
            cases = cases[: args.limit]
        report = run_evaluation(
            cases,
            modes=["dense", "hybrid"] if args.mode == "both" else [args.mode],
            retrieval_only=args.retrieval_only,
            with_ragas=args.ragas,
            include_text=args.include_text,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summaries"], indent=2))
    print(f"Report saved to {args.output}")
    for summary in report["summaries"].values():
        if (
            summary["run_errors"]
            or summary["unauthorized_chunks"]
            or summary["ragas_metric_errors"]
            or summary["input_guardrail_failures"]
        ):
            raise SystemExit(1)
        if not args.retrieval_only and summary["behavior_passes"] != summary["behavior_samples"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
