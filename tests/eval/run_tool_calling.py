#!/usr/bin/env python3
"""Measure how well a local model actually calls tools.

Not a pytest test: it needs a running Ollama server and a pulled model, so it
is run by hand when choosing a model or tuning tool selection.

    python tests/eval/run_tool_calling.py --model llama3.1:8b
    python tests/eval/run_tool_calling.py --model qwen3:8b --all-tools
    python tests/eval/run_tool_calling.py --model llama3.1:8b --compare

Which local model calls tools well moves faster than any list of
recommendations, and it depends on the machine. This answers the question on
the hardware that will run it.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from policy.paths import PathPolicy  # noqa: E402
from tests.eval.cases import CASES, EXTRA_TOOLS  # noqa: E402
from tools.local import apps, filesystem, web  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402
from tools.schema import Risk, ToolResult, ToolSpec  # noqa: E402
from tools.selection import ToolSelector, estimate_schema_tokens  # noqa: E402


def build_registry(with_extras: bool = True) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_all(filesystem.build_tools(PathPolicy(["/tmp"])))
    registry.register_all(web.build_tools())
    registry.register_all(apps.build_tools(["firefox", "code"]))
    if with_extras:
        for name, description in EXTRA_TOOLS:
            registry.register(ToolSpec(
                name=name,
                description=description,
                input_schema={"type": "object", "properties": {}},
                handler=lambda **kwargs: ToolResult("ok"),
                risk=Risk.WRITE,
            ))
    return registry


async def run(model: str, host: str, all_tools: bool, num_ctx: int, verbose: bool):
    import ollama

    from llm_module import LLMModule

    registry = build_registry()
    selector = ToolSelector(
        threshold=1 if not all_tools else 10_000,
        top_k=ToolSelector().top_k,
    )
    client = ollama.AsyncClient(host=host)

    correct = 0
    called_something = 0
    schema_tokens = []
    latencies = []
    failures = []

    for case in CASES:
        names = selector.select(registry, case.utterance, case.context)
        schemas = registry.describe(names)
        schema_tokens.append(estimate_schema_tokens(schemas))

        messages = [{"role": "system", "content": LLMModule.SYSTEM_PROMPT}]
        for line in (case.context or []):
            messages.append({"role": "user", "content": line})
            messages.append({"role": "assistant", "content": "D'accord."})
        messages.append({"role": "user", "content": case.utterance})

        started = time.monotonic()
        try:
            response = await client.chat(
                model=model,
                messages=messages,
                tools=schemas,
                options={"temperature": 0, "num_ctx": num_ctx},
            )
        except Exception as e:
            failures.append((case.utterance, f"request failed: {e}"))
            continue
        latencies.append(time.monotonic() - started)

        tool_calls = (response.get("message", {}) or {}).get("tool_calls") or []
        if not tool_calls:
            failures.append((case.utterance, f"no tool call (wanted {case.expected_tool})"))
            continue

        called_something += 1
        chosen = tool_calls[0]["function"]["name"]
        if chosen == case.expected_tool:
            correct += 1
            if verbose:
                print(f"  OK   {case.utterance[:50]:52} {chosen}")
        else:
            failures.append((case.utterance, f"called {chosen}, wanted {case.expected_tool}"))

    total = len(CASES)
    print(f"\n=== {model} ({'all tools' if all_tools else 'selected tools'}) ===")
    print(f"tools registered   : {len(registry.names())}")
    print(f"schema tokens      : ~{sum(schema_tokens) // max(1, len(schema_tokens))} per turn")
    print(f"called a tool      : {called_something}/{total} ({called_something / total:.0%})")
    print(f"called the RIGHT one: {correct}/{total} ({correct / total:.0%})")
    if latencies:
        print(f"median latency     : {sorted(latencies)[len(latencies) // 2]:.1f}s")

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for utterance, reason in failures:
            print(f"  {utterance[:50]:52} {reason}")

    return correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Ollama model, e.g. llama3.1:8b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument(
        "--all-tools", action="store_true",
        help="Offer every tool instead of the selected shortlist",
    )
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument(
        "--compare", action="store_true",
        help="Run both ways, to show what selection is worth",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.compare:
        selected = asyncio.run(
            run(args.model, args.host, False, args.num_ctx, args.verbose)
        )
        everything = asyncio.run(
            run(args.model, args.host, True, args.num_ctx, args.verbose)
        )
        print(f"\n=== selection is worth {selected - everything:+.0%} on {args.model} ===")
    else:
        asyncio.run(run(args.model, args.host, args.all_tools, args.num_ctx, args.verbose))


if __name__ == "__main__":
    main()
