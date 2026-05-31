from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .harness import Harness
from .runtime import RunState, ToolUseContext


class ToolExecutor:
    def __init__(self, harness: Harness):
        self.harness = harness

    def execute_tool_calls(self, tool_calls_parsed: list[tuple[Any, dict]],
                           state: RunState) -> list[tuple[Any, dict, Any]]:
        results: list[tuple[Any, dict, Any]] = []
        for batch in self.partition_tool_calls(tool_calls_parsed):
            if len(batch) > 1:
                results.extend(self._execute_parallel_batch(batch, state))
                continue

            tc, args = batch[0]
            result = self.harness.execute_tool(
                tc.function.name,
                args,
                context=ToolUseContext(
                    session_id=state.session_id,
                    messages=state.messages,
                    confirmed=False,
                ),
            )
            results.append((tc, args, result))
        return results

    def partition_tool_calls(self, tool_calls_parsed: list[tuple[Any, dict]]) -> list[list[tuple[Any, dict]]]:
        batches: list[list[tuple[Any, dict]]] = []
        for tc, args in tool_calls_parsed:
            tool = self.harness.tools.get(tc.function.name)
            concurrency_safe = bool(tool and tool.policy and tool.policy.concurrency_safe)
            if concurrency_safe and batches and self.batch_is_concurrency_safe(batches[-1]):
                batches[-1].append((tc, args))
            else:
                batches.append([(tc, args)])
        return batches

    def batch_is_concurrency_safe(self, batch: list[tuple[Any, dict]]) -> bool:
        return all(
            bool((tool := self.harness.tools.get(tc.function.name)) and tool.policy and tool.policy.concurrency_safe)
            for tc, _ in batch
        )

    def _execute_parallel_batch(self, batch: list[tuple[Any, dict]],
                                state: RunState) -> list[tuple[Any, dict, Any]]:
        with ThreadPoolExecutor(max_workers=min(len(batch), 4)) as pool:
            futures = {
                pool.submit(
                    self.harness.execute_tool,
                    tc.function.name,
                    args,
                    context=ToolUseContext(
                        session_id=state.session_id,
                        messages=state.messages,
                        confirmed=False,
                    ),
                ): (tc, args)
                for tc, args in batch
            }
            call_results = {}
            for future in futures:
                tc, args = futures[future]
                call_results[tc.id] = (tc, args, future.result())
        return [call_results[tc.id] for tc, _ in batch]
