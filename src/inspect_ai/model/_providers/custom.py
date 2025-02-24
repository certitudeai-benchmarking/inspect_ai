import requests
import json
from typing import Any
from inspect_ai.tool import ToolChoice, ToolInfo
from .._chat_message import ChatMessage
from .._generate_config import GenerateConfig
from .._model import ModelAPI
from .._model_output import ModelOutput, StopReason


class CustomModelAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any
    ) -> None:
        super().__init__(model_name, base_url, api_key, api_key_vars, config)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        url = self.base_url
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        messsages = [{"role": _input.role, "content": _input.content} for _input in input]
        payload = {
            "messages": messsages,
            "temperature": 0,
            "top_p": 0,
            "max_tokens": 800
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        stop_reason: StopReason | None = None
        try:
            output_json = response.json()
            content = output_json['choices'][0]['message']['content']
            stop_reason = "stop"
        except KeyError as e:
            stop_reason = "content_filter"
            content = str(e)
        except Exception as e:
            stop_reason = "unknown"
            content = str(e)
        return ModelOutput.from_content(self.model_name, content=content, stop_reason=stop_reason)
