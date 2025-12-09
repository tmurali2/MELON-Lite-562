from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
import time

import openai
from openai.types.chat import ChatCompletionReasoningEffort

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.openai_llm import _function_to_openai, _message_to_openai, _openai_to_assistant_message, chat_completion_request
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage

class OllamaLLM(BasePipelineElement):
    """LLM pipeline element that uses OpenAI's API.

    Args:
        client: The OpenAI client.
        model: The model name.
        temperature: The temperature to use for generation.
    """

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        reasoning_effort: ChatCompletionReasoningEffort | None = None,
        temperature: float | None = 0.0,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.reasoning_effort: ChatCompletionReasoningEffort | None = reasoning_effort

        self.log_path = Path("runs/ollama_llm_timings.log")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_path.open("a", encoding="utf-8")

        self._log(f"Starting log")

    def _log(self, msg: str):
        self.log_file.write(f"[{datetime.now(timezone.utc).isoformat()}] model={self.model} {msg}\n")
        self.log_file.flush()

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        openai_messages = [_message_to_openai(message, self.model) for message in messages]
        openai_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]
        
        start_time = time.time()
        completion = chat_completion_request(
            self.client, self.model, openai_messages, openai_tools, self.reasoning_effort, self.temperature
        )
        wall_ms = (time.time() - start_time) * 1000.0

        timings = getattr(completion, "timings", None)
        print(f"[ollama_llm] wall_ms: {wall_ms:.2f}ms, timings: {timings}")

        self._log(f"wall_ms={wall_ms:.2f} timings={timings}")

        output = _openai_to_assistant_message(completion.choices[0].message)
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args
