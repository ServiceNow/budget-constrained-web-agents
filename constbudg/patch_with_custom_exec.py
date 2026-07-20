import re
import time
import os

import playwright.sync_api

from browsergym.core.env import BrowserEnv, logger
from llm_client import get_client, llm_completion


agent_args = None


def _patch_webarena_openai():
    """
    Monkey-patch webarena's OpenAI client and LLM functions to use our llm_client instead.
    This allows us to route through OpenRouter and also log usage info.
    See here for more info: https://github.com/gasse/webarena/tree/main
    """
    try:
        from webarena.llms.providers import openai_utils
        
        def patched_get_openai_client():
            
            client, _ = get_client("openrouter/openai/gpt-4o")
            return client
        
        # This will use OpenRouter as it is called without specifying a prefix in the model name
        def patched_generate_from_openai_chat_completion(
            messages: list[dict[str, str]],
            model: str,
            temperature: float,
            max_tokens: int,
            top_p: float,
            context_length: int,
            stop_token: str | None = None,
        ) -> str:
            routed_model = os.environ.get("WEBARENA_EVAL_MODEL", "openrouter/openai/gpt-4o")  # gpt-4-1106-preview (default) is not currently available on OpenRouter

            kwargs = {
                "max_tokens": max_tokens,
                "top_p": top_p,
            }
            if stop_token:
                kwargs["stop"] = [stop_token]
            
            response = llm_completion(
                model=routed_model,
                messages=messages,
                temperature=temperature,
                usage_name="WebArena fuzzy_match",
                **kwargs
            )
            answer: str = response.choices[0].message.content
            return answer
        
        openai_utils.get_openai_client = patched_get_openai_client
        openai_utils.generate_from_openai_chat_completion = patched_generate_from_openai_chat_completion
        
    except ImportError:
        # webarena package not installed or different version
        pass


def _patch_webarena_task_timeout():
    """Override the default 10000ms timeout on the WebArena task class."""
    action_timeout_ms = int(
        os.environ.get(
            "PLAYWRIGHT_ACTION_TIMEOUT",
            os.environ.get("PLAYWRIGHT_NAVIGATION_TIMEOUT", "60000"),
        )
    )
    try:
        from browsergym.webarena.task import GenericWebArenaTask

        _original_init = GenericWebArenaTask.__init__

        def patched_init(self, *a, **kw):
            _original_init(self, *a, **kw)
            self.timeout = action_timeout_ms

        if not getattr(GenericWebArenaTask, "_timeout_patched", False):
            GenericWebArenaTask.__init__ = patched_init
            GenericWebArenaTask._timeout_patched = True
            logger.info(f"Patched WebArena task timeout to {action_timeout_ms}ms")
    except ImportError:
        pass


def execute_python_code(
    code: str,
    page: playwright.sync_api.Page,
    send_message_to_user: callable,
    report_infeasible_instructions: callable,
    **additional_globals
):
    """
    Executes Python code in a new context, except for a playwright `page` object and a `send_message_to_user` function.

    WARNING: this is not safe!
    https://stackoverflow.com/questions/77655440/can-you-protect-a-python-variable-with-exec

    Args:
        code: the Python code to execute, as a string.
        page: the playwright page that will be made accessible to the code.
        send_message_to_user: utility function that will be made accessible to the code. It should take one text argument.
        report_infeasible_instructions: utility function that will be made accessible to the code. It should take one text argument.
        additional_globals: additional global variables to make accessible to the code.
    """

    globals = {
        "page": page,
        "send_message_to_user": send_message_to_user,
        "report_infeasible_instructions": report_infeasible_instructions,
        **additional_globals,
    }

    exec(code, globals)


def step(self: BrowserEnv, action: str) -> tuple:
    """
    Small modification of the original BrowserEnv step method that uses the custom 
    execute_python_code function. 
    TODO: Probably better to refactor browsergym to support custom exec instead of this hack.
    """        
    self.last_action = action

    info = {}
    info["action_exec_start"] = time.time()
    info["action_exec_timeout"] = 0

    def send_message_to_user(text: str):
        if not isinstance(text, str):
            raise ValueError(f"Forbidden value: {text} is not a string")
        self.chat.add_message(role="assistant", msg=text)

    def report_infeasible_instructions(reason: str):
        if not isinstance(reason, str):
            raise ValueError(f"Forbidden value: {reason} is not a string")
        self.chat.add_message(role="infeasible", msg=reason)
        self.infeasible_message_received = True

    # try to execute the action
    logger.debug(f"Executing action")
    if not action:
        self.last_action_error = "No action provided (maybe due to LLM server error)"
    elif "```" not in action:
        self.last_action_error = "Action is not wrapped in triple backticks"
    else:
        try:
            if self.action_mapping:
                code = self.action_mapping(action)
            else:
                code = action
            execute_python_code(
                code,
                self.page,
                send_message_to_user=send_message_to_user,
                report_infeasible_instructions=report_infeasible_instructions,
                agent_args=agent_args,
                env=self,
            )
            self.last_action_error = ""
        except Exception as e:
            self.last_action_error = f"{type(e).__name__}: {e}"
            match = re.search(r"Timeout ([0-9]+)ms exceeded", self.last_action_error)  # Matches "Timeout 500ms exceeded" anywhere in the error string 
            if match:
                info["action_exec_timeout"] = float(match.groups()[0]) / 1000  # ms to sec
    logger.debug(f"Action executed")
    info["action_exec_stop"] = time.time()

    # wait a bit (for the JavaScript callback to set the active page)
    time.sleep(0.5)  # wait for JS events to be fired (half a second)
    self.context.cookies()  # trigger all waiting Playwright callbacks on the stack (hack, see https://playwright.dev/java/docs/multithreading)

    # wait for the network to idle before extracting the observation, reward etc.
    self._wait_dom_loaded()

    # after the action is executed, the active page might have changed
    # perform a safety check
    self._active_page_check()
    logger.debug(f"Active page checked")

    # if asked, wait for user message
    self._wait_for_user_message()
    logger.debug(f"User message done")

    logger.debug(f"Initiating task validation")
    # extract reward, done, user_message, info (task-specific)
    reward, done, user_message, task_info = self._task_validate()
    info["task_info"] = task_info
    logger.debug(f"Task validation done")

    # add any user message sent by the task to the chat
    if user_message:
        self.chat.add_message(role="user", msg=user_message)

    # extract observation (generic)
    obs = self._get_obs()
    logger.debug(f"Observation extracted")

    # new step API wants a 5-tuple (gymnasium)
    terminated = done or (
        self.terminate_on_infeasible and self.infeasible_message_received
    )  # task or agent can terminate the episode
    truncated = False

    return obs, reward, terminated, truncated, info


def _set_navigation_timeout(self: BrowserEnv, timeout_ms: int | None = None):
    """Set navigation and action timeouts on the browser context(s) and page."""
    timeout_ms = timeout_ms or int(
        os.environ.get("PLAYWRIGHT_NAVIGATION_TIMEOUT", "60000")
    )
    if hasattr(self, "browser") and self.browser and hasattr(self.browser, "contexts"):
        for ctx in self.browser.contexts:
            try:
                ctx.set_default_navigation_timeout(timeout_ms)
                ctx.set_default_timeout(timeout_ms)
            except Exception:
                pass
    if hasattr(self, "context") and self.context:
        try:
            self.context.set_default_navigation_timeout(timeout_ms)
            self.context.set_default_timeout(timeout_ms)
            logger.debug(f"Set context timeout to {timeout_ms}ms")
        except Exception as e:
            logger.warning(f"Failed to set context navigation timeout: {e}")
    if hasattr(self, "page") and self.page:
        try:
            self.page.set_default_navigation_timeout(timeout_ms)
            logger.debug(f"Set page navigation timeout to {timeout_ms}ms")
        except Exception as e:
            logger.warning(f"Failed to set page navigation timeout: {e}")



def reset(self: BrowserEnv, seed=None, options=None):
    """
    Patched reset method that sets navigation timeout to prevent timeouts during environment setup.
    This prevents timeouts during task setup (e.g., webarena ui_login page.goto calls).
    Since we patch Page.goto at the class level, we just need to ensure context timeout is set.
    """
    # Set timeout on existing context/page before reset (if they exist)
    self._set_navigation_timeout()
    
    # Call original reset method
    obs, info = self._original_reset(seed=seed, options=options)
    
    # After reset, set timeout again (in case new context/page were created)
    self._set_navigation_timeout()
    
    return obs, info


def patch_with_custom_exec(args):
    """Apply all patches: webarena OpenAI, WebArena task timeout, BrowserEnv step/reset, Page.goto timeout."""
    global agent_args
    agent_args = args

    _patch_webarena_openai()
    _patch_webarena_task_timeout()

    setattr(BrowserEnv, "step", step)
    
    # Bind timeout helper methods to BrowserEnv
    setattr(BrowserEnv, "_set_navigation_timeout", _set_navigation_timeout)
    
    # Patch Playwright Page.goto at the class level to ensure all pages use custom timeout
    if not hasattr(playwright.sync_api.Page, '_original_goto_patched'):
        original_goto = playwright.sync_api.Page.goto
        navigation_timeout_ms = int(os.environ.get("PLAYWRIGHT_NAVIGATION_TIMEOUT", "60000"))

        def patched_goto(self, url, **kwargs):
            # Use custom timeout if not specified
            if 'timeout' not in kwargs:
                kwargs['timeout'] = navigation_timeout_ms
            # Also ensure context and page timeouts are set
            if hasattr(self, 'context') and self.context:
                try:
                    self.context.set_default_navigation_timeout(navigation_timeout_ms)
                except Exception:
                    pass
            return original_goto(self, url, **kwargs)
        
        playwright.sync_api.Page.goto = patched_goto
        playwright.sync_api.Page._original_goto_patched = True
    
    # Store original reset method and replace it with patched version
    if not hasattr(BrowserEnv, '_original_reset'):
        BrowserEnv._original_reset = BrowserEnv.reset
    setattr(BrowserEnv, "reset", reset)