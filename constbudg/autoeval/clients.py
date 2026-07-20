import os
import base64
import openai
import numpy as np
from PIL import Image
from typing import Union, Optional
from openai import ChatCompletion
from llm_client import llm_completion, get_contents


class LM_Client:
    def __init__(self, model_name: str = "gpt-3.5-turbo") -> None:
        self.model_name = model_name

    def chat(self, messages, json_mode: bool = False) -> tuple[str, ChatCompletion]:
        """
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ])
        """
        # Use unified LLM client
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = llm_completion(
            model=self.model_name,
            messages=messages,
            temperature=0,
            usage_name=f"autoeval: LM Client",
            **kwargs
        )
        return get_contents(response, remove_thinking="qwen" in self.model_name)[0], response

    def one_step_chat(
        self, text, system_msg: str = None, json_mode=False
    ) -> tuple[str, ChatCompletion]:
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": text})
        return self.chat(messages, json_mode=json_mode)


class VLM_Client:
    def __init__(self, model_name: str = "gpt-4o", max_tokens: Optional[int] = None):
        self.model_name = model_name
        if max_tokens is None:
            max_tokens = 512 if "gpt" in self.model_name else 4096
        self.max_tokens = max_tokens

    def encode_image(self, path: str):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
                         
    def one_step_chat(
        self, text, image: Union[Image.Image, np.ndarray], 
        system_msg: Optional[str] = None,
    ) -> tuple[str, ChatCompletion]:
        jpg_base64_str = self.encode_image(image)
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages += [{
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{jpg_base64_str}"},},
                ],
        }]
        # Use unified LLM client
        response = llm_completion(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0,
            usage_name=f"autoeval: VLM Client",
        )
        return get_contents(response, remove_thinking="qwen" in self.model_name)[0], response


CLIENT_DICT = {
    "gpt-4o": VLM_Client,
    "qwen36_27b_vision": VLM_Client,
}