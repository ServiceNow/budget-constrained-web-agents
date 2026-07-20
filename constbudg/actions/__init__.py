# import the webarena default actions
import importlib
import os

from browsergym.core.action.functions import (
    click,  # click(elem)
    hover,  # hover(elem)
    fill,  # type(elem, text)
    keyboard_press,  # press(key_comb)
    scroll,  # scroll(dir)
    tab_focus,  # tab_focus(index)
    new_tab,  # new_tab()
    tab_close,  # tab_close()
    go_back,  # go_back()
    go_forward,  # go_forward()
    goto,  # goto(url)
    send_msg_to_user,  #
    report_infeasible,  # explicit unachievable action, equivalent to "N/A" answer
    select_option,  # select_option(elem, option)
)


def _import_site(site: str):
    suffix = os.environ.get("CONSTBUDG_RUN_SUFFIX", "default").strip() or "default"
    return importlib.import_module(f"actions.{suffix}.{site}")


# %% Import Induced Actions

shopping = _import_site("shopping")
admin = _import_site("admin")
reddit = _import_site("reddit")
map = _import_site("map")
gitlab = _import_site("gitlab")


def get_functions(module, prefix="actions.") -> list:
    functions = []
    for name in dir(module):
        try:
            f = getattr(module, name)
            if callable(f) and f.__module__.startswith(prefix):
                functions.append(f)
        except Exception:
            pass
    return functions


ACTION_DICT = {
    "webarena": [
        click, hover, fill, keyboard_press, scroll, tab_focus, new_tab,
        tab_close, go_back, go_forward, goto, send_msg_to_user, report_infeasible,
        select_option,
    ],
    "shopping": get_functions(shopping),
    "admin": get_functions(admin),
    "reddit": get_functions(reddit),
    "gitlab": get_functions(gitlab),
    "map": get_functions(map),
    "general": [],
}
