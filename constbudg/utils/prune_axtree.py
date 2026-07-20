#!/usr/bin/env python3
"""
Rule-based pruning of BrowserGym-style accessibility trees.

Remove StaticText nodes whose content is already present in the parent's accessible name (check immediate parent in hierarchy).
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


# Unicode private-use / icon characters (strip for comparison and from output)
PUA_PATTERN = re.compile(r"[\ue600-\ue6ff]|\\\\ue[0-9a-fA-F]{3}")


def strip_pua(s: str) -> str:
    r"""Remove private-use Unicode and literal \ueXXX from text."""
    if not s:
        return s
    # Literal backslash-u-hex in file
    s = re.sub(r"\\uE?[0-9a-fA-F]{3,4}", "", s)
    # Actual PUA codepoints
    return PUA_PATTERN.sub("", s).strip()


@dataclass
class AXLine:
    """One line of the axtree (parsed)."""
    indent: int
    bid: int | None
    role: str
    name: str
    rest: str
    raw: str
    leading: str = ""  # actual leading whitespace (tabs and/or spaces) from input

    @property
    def name_stripped(self) -> str:
        return strip_pua(self.name)

    def key_line(self) -> str:
        """Line as would be rendered (without children), with PUA stripped in name."""
        name_clean = strip_pua(self.name) or self.name
        quote = '"' if "'" in name_clean and '"' not in name_clean else "'"
        prefix = f"{' ' * self.indent}"
        if self.bid is not None:
            prefix += f"[{self.bid}] "
        return f"{prefix}{self.role} {quote}{name_clean}{quote}{self.rest}"


@dataclass
class AXNode:
    """Tree node built from axtree lines."""
    line: AXLine
    children: list["AXNode"] = field(default_factory=list)

    def structural_signature(self) -> tuple:
        """Signature for structure only (role + children signatures). Used to detect repeats."""
        child_sigs = tuple(c.structural_signature() for c in self.children)
        return (self.line.role, child_sigs)

    def all_same_structure(self, siblings: list["AXNode"]) -> bool:
        """True if all siblings have the same structural signature as self."""
        sig = self.structural_signature()
        return all(s.structural_signature() == sig for s in siblings)


def parse_line(line: str) -> AXLine | None:
    """Parse one axtree line. Returns None for blank/unparseable."""
    stripped = line.rstrip("\n")
    if not stripped:
        return None
    leading = stripped[: len(stripped) - len(stripped.lstrip())]
    indent = len(leading)
    rest = stripped.lstrip()
    # Optional [id]
    bid = None
    m = re.match(r"^\[(\d+)\]\s+", rest)
    if m:
        bid = int(m.group(1))
        rest = rest[m.end() :]
    # Type (one or more tokens until space + quote)
    type_end = rest.find(" '")
    if type_end == -1:
        type_end = rest.find(' "')
    if type_end == -1:
        return AXLine(indent=indent, bid=bid, role="", name="", rest=rest, raw=stripped, leading=leading)
    role = rest[:type_end].strip()
    rest = rest[type_end + 1 :]
    quote = rest[0]
    # Find closing quote (no escaping in samples)
    close = 1
    while close < len(rest):
        i = rest.find(quote, close)
        if i == -1:
            break
        close = i + 1
        break
    else:
        close = len(rest)
    name = rest[1 : close - 1]
    rest = rest[close:].lstrip()
    if rest and not rest.startswith(","):
        rest = " " + rest
    return AXLine(indent=indent, bid=bid, role=role, name=name, rest=rest, raw=stripped, leading=leading)


def _detect_indent_unit(parsed: list[AXLine]) -> str:
    """Detect indent string from first line that has leading whitespace (preserve tabs vs spaces)."""
    for p in parsed:
        if p.leading:
            return p.leading
    return "    "  # default 4 spaces

def parse_axtree(text: str) -> tuple[list[AXLine], list[AXNode], str]:
    """Parse full axtree text into lines, tree, and indent_unit (for rendering)."""
    lines = text.split("\n")
    parsed: list[AXLine] = []
    for line in lines:
        pl = parse_line(line)
        if pl is not None:
            parsed.append(pl)
    if not parsed:
        return parsed, [], "    "
    indent_unit = _detect_indent_unit(parsed)
    min_indent = min(p.indent for p in parsed)
    nodes_by_i: list[AXNode | None] = [None] * len(parsed)
    for i, p in enumerate(parsed):
        nodes_by_i[i] = AXNode(line=p)
    stack: list[int] = []
    for i, p in enumerate(parsed):
        while stack and parsed[stack[-1]].indent >= p.indent:
            stack.pop()
        if stack:
            parent_i = stack[-1]
            nodes_by_i[parent_i].children.append(nodes_by_i[i])
        stack.append(i)
    roots = [nodes_by_i[i] for i, p in enumerate(parsed) if p.indent == min_indent]
    return parsed, roots, indent_unit


TEXT_SEMANTIC_ROLES = {
    "article",
    "paragraph",
    "heading",
    "strong",
    "emphasis",
    "mark",
    "sectionheader",
}


def should_remove_static_text(
    node: AXNode,
    parent_name: str,
    parent_role: str = "",
) -> bool:
    """True if this StaticText's content is already in the parent's name.

    StaticText inside semantic text containers (headings, paragraphs, ...) is
    always kept to preserve the readable text of the page.
    """
    if node.line.role != "StaticText":
        return False
    child_text = node.line.name_stripped
    if not child_text:
        return True
    if parent_role in TEXT_SEMANTIC_ROLES:
        return False
    return child_text in parent_name


def prune_static_text_roots(roots: list[AXNode]) -> None:
    """In-place: remove StaticText children whose content is in parent's name."""
    def recurse(node: AXNode, parent_name: str) -> None:
        parent_name_for_children = node.line.name_stripped
        kept: list[AXNode] = []
        for c in node.children:
            if should_remove_static_text(
                c,
                parent_name_for_children,
                parent_role=node.line.role,
            ):
                continue
            kept.append(c)
            recurse(c, parent_name_for_children)
        node.children[:] = kept
    for r in roots:
        recurse(r, "")


def render_node(node: AXNode, indent: int, indent_unit: str) -> list[str]:
    """Render one node and its children."""
    out: list[str] = []
    # Preserve tabs vs spaces: depth from our internal indent (0,4,8,...)
    depth = indent // 4
    prefix = indent_unit * depth
    line_str = node.line.key_line().lstrip()
    out.append(f"{prefix}{line_str}")
    for c in node.children:
        out.extend(render_node(c, indent + 4, indent_unit))
    return out


def render_roots(roots: list[AXNode], indent_unit: str = "    ") -> list[str]:
    """Render tree to lines. Uses key_line for consistent spacing."""
    out: list[str] = []
    for r in roots:
        out.extend(render_node(r, r.line.indent, indent_unit))
    return out


def prune_axtree(text: str) -> str:
    """Full pipeline: parse, prune StaticText, render."""
    parsed, roots, indent_unit = parse_axtree(text)
    if not roots:
        return text.strip()
    prune_static_text_roots(roots)
    lines = render_roots(roots, indent_unit=indent_unit)
    return "\n".join(lines)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Prune BrowserGym-style axtree: remove redundant StaticText.")
    ap.add_argument("input", nargs="?", type=Path, default=None, help="Input axtree file (default: stdin)")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output file (default: stdout)")
    args = ap.parse_args()
    src = Path(args.input).read_text() if args.input else sys.stdin.read()
    out = prune_axtree(src)
    if args.output:
        args.output.write_text(out + "\n")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
