import base64
import dataclasses
import io
import os
import logging
import re

import numpy as np
from PIL import Image

from browsergym.experiments import AbstractAgentArgs, Agent
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html

from custom_action_set import CustomActionSet
from actions import ACTION_DICT
from llm_client import llm_completion, get_contents
from utils.prune_axtree import prune_axtree

logger = logging.getLogger(__name__)

# Store latest token usage
_latest_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

_FILL_LINE_RE = re.compile(r"""^(\s*)fill\((['"])([^'"]+)\2\s*,.*\)\s*$""")
_CLICK_BID_LINE_RE = re.compile(r"""^(\s*)click\((['"])([^'"]+)\2\)\s*$""")


def _patch_fill_click_before_block(code: str) -> str:
    """Insert click(bid) immediately before each fill(bid, ...)."""
    lines = code.splitlines()
    out: list[str] = []
    for line in lines:
        m_fill = _FILL_LINE_RE.match(line)
        if m_fill:
            indent, quote, bid = m_fill.group(1), m_fill.group(2), m_fill.group(3)
            already_clicked = False
            for prev in reversed(out):
                if not prev.strip():
                    continue
                m_click = _CLICK_BID_LINE_RE.match(prev)
                already_clicked = m_click is not None and m_click.group(3) == bid
                break
            if not already_clicked:
                out.append(f"{indent}click({quote}{bid}{quote})")
        out.append(line)
    return "\n".join(out)


def patch_gitlab_action(action: str) -> str:
    """Rewrite agent actions for GitLab focus issues on text inputs. On GitLab, the global 
    search box changes when it is focused, and this change leads to a new BID. However, 
    `fill(old_bid, text)` has an issue and hangs instead of raising an error and communicating 
    the error to the user. We manually add a no-effect `click(old_bid)` before `fill(old_bid, text)` 
    to make the text box focused which makes the fill action raise an error and not hang.
    """
    if not action or "```" not in action:
        return action

    parts = action.split("```")
    for idx in range(1, len(parts), 2):
        parts[idx] = _patch_fill_click_before_block(parts[idx])
    return "```".join(parts)


def _active_page_url(obs: dict) -> str:
    urls = obs.get("open_pages_urls") or []
    if not urls:
        return ""
    idx = int(np.asarray(obs.get("active_page_index", 0)).item())
    return urls[min(idx, len(urls) - 1)]


def _is_gitlab_context(obs: dict, websites: tuple[str, ...]) -> bool:
    try:
        if "gitlab" in websites:
            return True
        url = _active_page_url(obs)
        if not url:
            return False
        gitlab_base = (os.environ.get("WA_GITLAB") or "").strip().rstrip("/")
        if gitlab_base and gitlab_base.lower() != "todo" and url.startswith(gitlab_base):
            return True
        return "gitlab" in url.lower()
    except:
        return False


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


class DemoAgent(Agent):
    """A basic agent using OpenAI API, to demonstrate BrowserGym's functionalities."""

    def _dump_axtree_snapshot(self, axtree_txt: str, pruned_axtree_txt: str) -> None:
        if not self.axtree_dump_dir:
            return

        os.makedirs(self.axtree_dump_dir, exist_ok=True)
        step_name = f"step_{self.axtree_dump_index:03d}"
        raw_path = os.path.join(self.axtree_dump_dir, f"{step_name}_axtree.txt")
        pruned_path = os.path.join(self.axtree_dump_dir, f"{step_name}_pruned_axtree.txt")

        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(axtree_txt)
        with open(pruned_path, "w", encoding="utf-8") as f:
            f.write(pruned_axtree_txt)

        self.axtree_dump_index += 1

    def obs_preprocessor(self, obs: dict) -> dict:
        axtree_txt = flatten_axtree_to_str(obs["axtree_object"])
        pruned_axtree_txt = prune_axtree(axtree_txt)
        self._dump_axtree_snapshot(axtree_txt, pruned_axtree_txt)
        return {
            "chat_messages": obs["chat_messages"],
            "screenshot": obs["screenshot"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "open_pages_urls": obs["open_pages_urls"],
            "open_pages_titles": obs["open_pages_titles"],
            "active_page_index": obs["active_page_index"],
            "axtree_txt": axtree_txt,
            "pruned_axtree_txt": pruned_axtree_txt,
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
        }

    def __init__(
        self,
        model_name: str,
        chat_mode: bool,
        demo_mode: str,
        use_html: bool,
        use_axtree: bool,
        use_screenshot: bool,
        websites: tuple[str],
        actions: list[str],
        memory: str,
        temperature: float = 0.0,
        prune_axtree: bool = False,
        axtree_dump_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.chat_mode = chat_mode
        self.use_html = use_html
        self.use_axtree = use_axtree
        self.use_screenshot = use_screenshot
        self.temperature = temperature
        self.prune_axtree = prune_axtree
        self.axtree_dump_dir = axtree_dump_dir
        self.axtree_dump_index = 0
        s = ""
        if self.prune_axtree:
            s += "Pruning the axtree before sending to the LLM. "
        if self.axtree_dump_dir:
            s += f" Saving raw and pruned axtrees to {self.axtree_dump_dir}."
        if s:
            logger.info(s)

        if not (use_html or use_axtree):
            raise ValueError(f"Either use_html or use_axtree must be set to True.")

        self.websites = tuple(websites)

        custom_actions = ACTION_DICT["general"] + ACTION_DICT["webarena"]
        for website in websites:
            custom_actions.extend(ACTION_DICT[website])

        self.action_set = CustomActionSet(
            subsets=["custom"],
            custom_actions=custom_actions,
            strict=False,  # less strict on the parsing of the actions
            multiaction=True,  # enable the agent to take multiple actions at once
            demo_mode=demo_mode,  # add visual effects
        )

        self.action_history = []

        self.actions = actions
        self.num_actions = 0
        
        if memory is None: self.memory = None
        else: 
            paths = memory.split(' ')
            self.memory = '\n\n'.join([open(p, 'r').read() for p in paths])
            if self.memory.strip() == "":
                self.memory = None
        
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        global _latest_token_usage
        _latest_token_usage = self.token_usage

    def get_action(self, obs: dict) -> tuple[str, dict]:
        if len(self.actions) == 0 or (self.num_actions > (len(self.actions) - 1)):
            system_msgs = []
            user_msgs = []

            if self.chat_mode:
                system_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
    # Instructions

    You are a UI Assistant, your goal is to help the user perform tasks using a web browser. You can
    communicate with the user via a chat, to which the user gives you instructions and to which you
    can send back messages. You have access to a web browser that both you and the user can see,
    and with which only you can interact via specific commands.

    Review the instructions from the user, the current state of the page and all other information
    to find the best possible next action to accomplish your goal. Your answer will be interpreted
    and executed by a program, make sure to follow the formatting instructions.
    """,
                    }
                )

                # append chat messages
                user_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
    # Chat Messages
    """,
                    }
                )
                for msg in obs["chat_messages"]:
                    if msg["role"] in ("user", "assistant", "infeasible"):
                        user_msgs.append(
                            {
                                "type": "text",
                                "text": f"""\
    - [{msg['role']}] {msg['message']}
    """,
                            }
                        )
                    elif msg["role"] == "user_image":
                        user_msgs.append({"type": "image_url", "image_url": msg["message"]})
                    else:
                        raise ValueError(f"Unexpected chat message role {repr(msg['role'])}")

            else:
                assert obs["goal_object"], "The goal is missing."
                system_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
    # Instructions

    Review the current state of the page and all other information to find the best
    possible next action to accomplish your goal. Your answer will be interpreted
    and executed by a program, make sure to follow the formatting instructions.
    """,
                    }
                )
                # append memory
                if self.memory is not None:
                    system_msgs.append({
                        "type": "text",
                        "text": self.memory,
                    })
                # append goal
                user_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
    # Goal
    """,
                    }
                )
                # goal_object is directly presented as a list of openai-style messages
                user_msgs.extend(obs["goal_object"])

            # append url of all open tabs
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
    # Currently open tabs
    """,
                }
            )
            for page_index, (page_url, page_title) in enumerate(
                zip(obs["open_pages_urls"], obs["open_pages_titles"])
            ):
                user_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
    Tab {page_index}{" (active tab)" if page_index == obs["active_page_index"] else ""}
    Title: {page_title}
    URL: {page_url}
    """,
                    }
                )

            # append page AXTree (if asked)
            if self.use_axtree:
                user_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
    # Current page Accessibility Tree

    {obs["pruned_axtree_txt"] if self.prune_axtree else obs["axtree_txt"]}

    """,
                    }
                )
            # append page HTML (if asked)
            if self.use_html:
                user_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
    # Current page DOM

    {obs["pruned_html"]}

    """,
                    }
                )

            # append page screenshot (if asked)
            if self.use_screenshot:
                user_msgs.append(
                    {
                        "type": "text",
                        "text": """\
    # Current page Screenshot
    """,
                    }
                )
                user_msgs.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_jpg_base64_url(obs["screenshot"]),
                            "detail": "auto",
                        },  # Literal["low", "high", "auto"] = "auto"
                    }
                )
        
            # append action space description
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
    # Action Space

    {self.action_set.describe(with_long_description=True, with_examples=True)}

    When high-level functions such as `get_driving_time` or `book_flights` are available, please prioritize using them.
    Here are examples of actions with chain-of-thought reasoning:

    I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
    ```click("12")```

    I found the information requested by the user, I will send it to the chat.
    ```send_msg_to_user('The price for a 15" laptop is 1499 USD.')```

    You must wrap the to-be-executed action in triple backticks (like this: '```click(\"12\")```'). Do not wrap the reasoning or the action description.
    Do not use triple backticks anywhere else as the runtime parses the first one only.
    Multiple actions (if needed) should be wrapped in the same triple backticks block. Each action must be a single line; use normal Python quoting so long strings stay on one line.

    """,
                }
            )

            # append past actions (and last error message) if any
            if self.action_history:
                user_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
    # History of past actions
    """,
                    }
                )
                user_msgs.extend(
                    [
                        {
                            "type": "text",
                            "text": f"""\

    {action}
    """,
                        }
                        for action in self.action_history
                    ]
                )

                if obs["last_action_error"]:
                    user_msgs.append(
                        {
                            "type": "text",
                            "text": f"""\
    # Error message from last action

    {obs["last_action_error"]}

    """,
                        }
                    )
                    print("Error message from last action: ", obs["last_action_error"])

            # ask for the next action
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
    # Next action

    You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, and the current state of the page before deciding on your next action.
    """,
                }
            )

            prompt_text_strings = []
            for message in system_msgs + user_msgs:
                match message["type"]:
                    case "text":
                        prompt_text_strings.append(message["text"])
                    case "image_url":
                        image_url = message["image_url"]
                        if isinstance(message["image_url"], dict):
                            image_url = image_url["url"]
                        if image_url.startswith("data:image"):
                            prompt_text_strings.append(
                                "image_url: " + image_url[:30] + "... (truncated)"
                            )
                        else:
                            prompt_text_strings.append("image_url: " + image_url)
                    case _:
                        raise ValueError(
                            f"Unknown message type {repr(message['type'])} in the task goal."
                        )

            try:
                if "gpt-5" in self.model_name:
                    user_msgs.append({"type": "text", "text": """
    You cannot ask for follow-up questions or hand back to the user. Do your best with the information already available.
    Output a summary of your reasoning and the important information you have found, followed by the best actions at this step (in one triple backticks block)."""})

                response = llm_completion(  # usage_name is not provided here since in DemoAgent we print the accumulated token usage at the end
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_msgs},
                        {"role": "user", "content": user_msgs},
                    ],
                    temperature=self.temperature,
                )
                action = get_contents(response, placeholder_if_failed="")[0]
                if "qwen" in self.model_name:  # Qwen's thinking might contain ```, so we clean the output first
                    if "</think>" in action:
                        splits = action.split("</think>")
                        reasoning = "\n".join(splits[:-1])
                        answer = splits[-1]
                        if len(answer.split("```")) > 3:
                            answer = answer.replace("```\n```", "\n")
                        action = reasoning.replace("```", "") + "</think>" + answer
                    else:
                        answer = action.split("```")[-2]
                        action = action.replace("```", "") + "\n```" + answer + "```"
                action = action.replace('```python', '```')
                if _is_gitlab_context(obs, self.websites):
                    action = patch_gitlab_action(action)
                if hasattr(response, 'usage') and response.usage:
                    self.token_usage["prompt_tokens"] += getattr(response.usage, 'prompt_tokens', 0) or 0
                    self.token_usage["completion_tokens"] += getattr(response.usage, 'completion_tokens', 0) or 0
                    self.token_usage["total_tokens"] += getattr(response.usage, 'total_tokens', 0) or 0
            except Exception as e:
                logging.error(f"Error in LLM completion: {e}")
                action = ""
        else:
            if self.num_actions > (len(self.actions) - 1):
                action = None
            else:
                action = self.actions[self.num_actions]
            self.num_actions += 1

        self.action_history.append(action)

        return action, {}




@dataclasses.dataclass
class DemoAgentArgs(AbstractAgentArgs):
    """
    This class is meant to store the arguments that define the agent.

    By isolating them in a dataclass, this ensures serialization without storing
    internal states of the agent.
    """

    model_name: str = "openrouter/google/gemini-3-flash-preview"
    chat_mode: bool = False
    demo_mode: str = "off"
    use_html: bool = False
    use_axtree: bool = True
    use_screenshot: bool = False
    websites: tuple[str] = ()
    actions: list[str] = ()
    memory: str = None
    temperature: float = 0.0
    prune_axtree: bool = False
    axtree_dump_dir: str | None = None

    def make_agent(self):
        return DemoAgent(
            model_name=self.model_name,
            chat_mode=self.chat_mode,
            demo_mode=self.demo_mode,
            use_html=self.use_html,
            use_axtree=self.use_axtree,
            use_screenshot=self.use_screenshot,
            websites=self.websites,
            actions=self.actions,
            memory=self.memory,
            temperature=self.temperature,
            prune_axtree=self.prune_axtree,
            axtree_dump_dir=self.axtree_dump_dir,
        )
    
    def get_and_reset_token_usage(self):
        """Get token usage from the most recent agent instance."""
        global _latest_token_usage
        token_usage = _latest_token_usage.copy()
        _latest_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return token_usage
