import inspect
import random
import re
import browsergym.core.action.utils as utils
import pyparsing as pp
import torch
from sentence_transformers import SentenceTransformer

from typing import Optional
from browsergym.core.action.functions import noop
from browsergym.core.action.highlevel import (
    ACTION_SUBSETS,
    HighLevelActionSet,
)

from dataclasses import dataclass
@dataclass
class HighLevelAction:
    # entrypoint: callable
    signature: str
    description: str
    examples: list[str]
    function: str
    add_code: bool

from parsers import _build_python_subset_parser

# Parser for extracting python-like code
python_subset_parser: pp.ParserElement = _build_python_subset_parser()

class CustomActionSet(HighLevelActionSet):

    def __init__(
        self,
        subsets: Optional[HighLevelActionSet.ActionSubset | list[HighLevelActionSet.ActionSubset]] = [
            "chat",
            "infeas",
            "bid",
            "nav",
            "tab",
        ],
        custom_actions: Optional[list[callable]] = None,
        multiaction: bool = True,
        demo_mode: Optional[HighLevelActionSet.DemoMode] = None,
        strict: bool = False,
        retry_with_force: bool = False,
        retrievable_actions: Optional[list[str]] = None,
        retrieval_model_name: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    ):
        self.strict = strict
        self.multiaction = multiaction
        self.demo_mode = demo_mode
        self.retry_with_force = retry_with_force
        if retrievable_actions is None:
            retrievable_actions = []
        self.retrieval_model_name = retrieval_model_name

        if not subsets:
            raise ValueError(f"'action_subsets' is empty.")

        if isinstance(subsets, str):
            subsets = [subsets]

        allowed_actions = [noop]  # the noop action is always allowed

        # add actions from specified action sets
        if subsets:
            for subset in subsets:
                if subset in ACTION_SUBSETS:
                    allowed_actions.extend(ACTION_SUBSETS[subset])
                elif subset == "custom":
                    if not custom_actions:
                        raise ValueError(
                            "'custom' is in 'action_subsets' but 'custom_actions' is empty."
                        )
                    allowed_actions.extend(custom_actions)
                else:
                    raise ValueError(f"Unknown high-level action subspace: {subset}")

        # like set() but preserves order
        # https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set
        allowed_actions = list(dict.fromkeys(allowed_actions).keys())
        retrievable_actions = list(dict.fromkeys(retrievable_actions).keys())

        # parse the actions and build the action space
        self.action_set: dict[str, HighLevelAction] = {}
        self.retrievable_action_set: dict[str, HighLevelAction] = {}
        self.python_includes = ""

        # include playwright and browsergym imports
        self.python_includes += f"""\
import playwright.sync_api
import requests
from typing import Literal
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from bs4 import BeautifulSoup


"""
        # set demo_mode and retry_with_force flags
        self.python_includes += f"""\
demo_mode={repr(demo_mode)}
retry_with_force={repr(retry_with_force)}

if demo_mode is None:
    demo_mode = "default" if DEMO_MODE else "off"

"""

        # include utility functions
        for _, func in inspect.getmembers(utils, inspect.isfunction):
            self.python_includes += f"""\
{inspect.getsource(func)}


"""

        self._parse_and_include_actions(allowed_actions, self.action_set)
        self._parse_and_include_actions(retrievable_actions, self.retrievable_action_set)

        if retrievable_actions:
            self._build_retrieval_index()
    

    def _build_retrieval_index(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else self.device)
        self.retrieval_model = SentenceTransformer(self.retrieval_model_name, trust_remote_code=True).to(self.device)
        self.retrieval_max_length = 512
        action_descriptions = [self.get_action_doc(action, with_long_description=True, with_examples=True) for action in self.retrievable_action_set.values()]
        self.action_embeddings = self.retrieval_model.encode(action_descriptions, instruction="", max_length=self.retrieval_max_length, convert_to_tensor=True)
        
    def retrieve_actions(self, query: str, num_retrieve: int) -> dict[str, HighLevelAction]:
        instruction = "Instruct: Given a user query, retrieve functions that could be useful to fulfill the query.\nQuery: "
        query_embedding = self.retrieval_model.encode(query, instruction=instruction, max_length=self.retrieval_max_length, convert_to_tensor=True)
        scores = (query_embedding @ self.action_embeddings.T).squeeze()
        assert len(scores) == len(self.retrievable_action_set)
        top_actions = scores.argsort(descending=True)[:num_retrieve]
        return {list(self.retrievable_action_set.keys())[i]: list(self.retrievable_action_set.values())[i] for i in top_actions}
        
            
    def _parse_and_include_actions(self, allowed_actions, action_set):
                # parse and include action functions
        for func in allowed_actions:

            # include action function definition in the code
            self.python_includes += f"""\
{inspect.getsource(func)}


"""
            # extract action signature
            signature = f"{func.__name__}{inspect.signature(func)}"

            # parse docstring
            m = re.search(r"(Examples?:)", func.__doc__)
            if m:
                description, examples = func.__doc__[:m.start()].strip(), func.__doc__[m.end():].strip()
            else:
                description, examples = func.__doc__.strip(), ""

            if func.__name__ in self.action_set:
                raise ValueError(f"Duplicated action '{func.__name__}'")

            action_file = inspect.getfile(func)
            add_code = ("/actions/" in action_file) and ("__init__.py" not in action_file)
            action_set[func.__name__] = HighLevelAction(
                # entrypoint=func,
                signature=signature,
                description=description,
                examples=[examples.strip()],
                function=inspect.getsource(func),
                add_code=add_code,
            )
    
    def example_action(self, abstract: bool, max_examples: int = 3) -> str:
        """
        Returns an example action as a string.
        """
        if abstract:
            if self.multiaction:
                return """\
One or several actions, separated by new lines."""
            else:
                return """\
One single action to be executed. You can only use one action at a time."""
        else:
            picked_examples = []

            # use fill and click examples if action is present
            for action_name in ["fill", "click", "mouse_click", "keyboard_type"]:
                if action_name in self.action_set:
                    picked_examples.extend(self.action_set[action_name].examples)

            # last resort, use all action examples
            if not picked_examples:
                for _, action in self.action_set.items():
                    picked_examples += action.examples

            # shuffle examples
            rng = random.Random(1)
            rng.shuffle(picked_examples)

            if self.multiaction:
                return "\n".join(picked_examples[:max_examples])
            else:
                return picked_examples[0]

    @staticmethod
    def get_action_doc(action: HighLevelAction, with_long_description: bool=True, with_examples=True) -> str:
        """"
        Returns the documentation of the given action/tool.
        """
        if with_long_description and with_examples and action.add_code:
            description = f"""\
{action.function}
"""
        else:  
            description = f"""\
{action.signature}
"""
            if with_long_description:
                description += f"""\
    Description: {action.description}
"""
            if with_examples and action.examples:
                description += f"""\
    Examples:
"""
                for example in action.examples:
                    description += f"""\
        {example}

"""
        return description

            
    def describe(self, with_long_description: bool = True, with_examples: bool = True, retrieval_query: Optional[str] = None, num_retrieve: int=0) -> str:
        """
        Returns a textual description of this action space.
        """
        action_set = self.action_set
        if retrieval_query and num_retrieve and self.retrievable_action_set:
            retrieved_actions = self.retrieve_actions(retrieval_query, num_retrieve)
            action_set = {**action_set, **retrieved_actions}
        
        description = f"""
{len(action_set)} different types of actions are available.

"""
        for _, action in action_set.items():
            description += self.get_action_doc(action, with_long_description, with_examples)

        if self.multiaction:
#             description += f"""\
# Multiple actions can be provided at once. An action can consume the output of a previous action by using the output variable. Though, note when using multiple actions, you may find it helpful to leverage the `observe` function if you need to query the state of the page between actions."""
            description += f"""\
Multiple actions can be provided at once. An action can consume the output of a previous action by using the output variable."""
        else:
            description += f"""\
Only a single action can be provided at once."""

        return description

    def to_python_code(self, action):
        """
        Converts the given high-level action string to browsergym-compatible python code.

        Args:
            action: the high-level action to parse.

        Returns:
            Executable python code that performs the action in a browsergym environment.
        """
        program = action.split("```")[1]
        
        # Make sure we can parse generated code
        python_subset_parser.parse_string(program)
        
        python_code = ""

        # function definitions
        python_code += self.python_includes

        # return the constructed python code
        return python_code + program
