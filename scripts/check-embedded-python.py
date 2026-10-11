#!/usr/bin/env python3
"""Regression check for #88: every Python template embedded in LTXBridge.swift
must render to syntactically valid Python.

Swift multi-line strings process escapes (backslash-n becomes a real
newline) and backslash-paren interpolation at compile time. A single
backslash-n inside an embedded
Python string literal therefore splits it across two lines, and the whole
generated script dies with `SyntaxError: unterminated string literal` at parse
time -- before any of its logic runs. That is exactly what broke all LTX-2.5
generation in #88 (Swift line 1170 rendered Python line 166).

This script replays Swift's rendering (escape processing, closing-delimiter
indentation stripping, interpolation substitution) over the real
LTXBridge.swift content and `ast.parse`s each discovered template. Templates
are discovered dynamically, so the check keeps working as templates move.

Usage:
    scripts/check-embedded-python.py [path/to/LTXBridge.swift]
Exit 0 when every embedded Python template parses, 1 otherwise.
"""
import ast
import sys
from pathlib import Path

DEFAULT_BRIDGE = (
    Path(__file__).resolve().parent.parent
    / "LTXVideoGenerator"
    / "Sources"
    / "Services"
    / "LTXBridge.swift"
)


def discover_templates(lines):
    """Yield (name, open_lineno, close_lineno) 1-based per string block."""
    i = 0
    while i < len(lines):
        stripped = lines[i].rstrip()
        if stripped.endswith('= """') and '"""' not in stripped[:-4].strip():
            opener = lines[i]
            name = opener.strip().split("=")[0].strip().split()[-1]
            j = i + 1
            while j < len(lines) and lines[j].strip() != '"""':
                j += 1
            if j >= len(lines):
                raise SystemExit(
                    f"error: unterminated Swift string opened at line {i + 1}"
                )
            yield name, i + 1, j + 1
            i = j + 1
        else:
            i += 1


def render(body_lines, first_lineno, closing_indent):
    """Simulate Swift multiline-string rendering.

    Returns (text, line_map, problems). line_map[k] is the absolute Swift
    source line for rendered line k+1. problems lists (swift_line, detail)
    for anything Swift would reject or this renderer cannot model.
    """
    rendered_lines = []
    rendered_map = []
    problems = []
    for lineno, raw in zip(
        range(first_lineno, first_lineno + len(body_lines)), body_lines
    ):
        if raw.strip():
            if raw.startswith(closing_indent):
                raw = raw[len(closing_indent) :]
            else:
                problems.append((lineno, "insufficient indentation"))
                continue
        segments = [[]]
        j = 0
        while j < len(raw):
            ch = raw[j]
            if ch != "\\":
                segments[-1].append(ch)
                j += 1
                continue
            if j + 1 >= len(raw):
                problems.append((lineno, "trailing backslash"))
                j += 1
                continue
            nxt = raw[j + 1]
            if nxt == "n":
                segments.append([])  # Swift `\n` -> real newline
            elif nxt == "t":
                segments[-1].append("\t")
            elif nxt == "r":
                segments[-1].append("\r")
            elif nxt == "0":
                segments[-1].append("\0")
            elif nxt == "\\":
                segments[-1].append("\\")
            elif nxt in ('"', "'"):
                segments[-1].append(nxt)
            elif nxt == "u" and raw[j + 2 : j + 3] == "{":
                end = raw.find("}", j + 3)
                if end == -1:
                    problems.append((lineno, "bad \\u escape"))
                    j += 2
                    continue
                segments[-1].append(chr(int(raw[j + 3 : end], 16)))
                j = end + 1
                continue
            elif nxt == "(":
                # Swift interpolation: skip balanced parens (string-aware),
                # substitute a parse-neutral token.
                depth = 1
                k = j + 2
                in_str = False
                while k < len(raw) and depth > 0:
                    c = raw[k]
                    if in_str:
                        if c == "\\":
                            k += 2
                            continue
                        if c == '"':
                            in_str = False
                        k += 1
                        continue
                    if c == '"':
                        in_str = True
                    elif c == "(":
                        depth += 1
                    elif c == ")":
                        depth -= 1
                    k += 1
                if depth != 0:
                    problems.append((lineno, "unbalanced interpolation"))
                    j = k
                    continue
                segments[-1].append("X")
                j = k
                continue
            else:
                problems.append((lineno, f"unknown escape \\{nxt}"))
                segments[-1].append(ch)
                j += 1
                continue
            j += 2
        for seg in segments:
            rendered_lines.append("".join(seg))
            rendered_map.append(lineno)
    return "\n".join(rendered_lines) + "\n", rendered_map, problems


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BRIDGE
    lines = path.read_text().splitlines()
    failures = 0
    checked = 0
    for name, open_ln, close_ln in discover_templates(lines):
        body = lines[open_ln : close_ln - 1]  # 0-based slice of template lines
        first = next((ln for ln in body if ln.strip()), "")
        if not first.lstrip().startswith(("import ", "from ")):
            continue  # Swift-side prose, not an embedded Python script
        checked += 1
        closing_line = lines[close_ln - 1]
        closing_indent = closing_line[: len(closing_line) - len(closing_line.lstrip())]
        rendered, line_map, problems = render(body, open_ln + 1, closing_indent)
        if problems:
            print(f"[{name}] (Swift lines {open_ln}-{close_ln}) render problems:")
            for lineno, detail in problems:
                print(f"    Swift line {lineno}: {detail}")
            failures += 1
            continue
        try:
            ast.parse(rendered)
            print(f"[{name}] parses OK ({len(rendered.splitlines())} lines)")
        except SyntaxError as e:
            swift_line = (
                line_map[e.lineno - 1]
                if e.lineno and e.lineno - 1 < len(line_map)
                else "?"
            )
            print(
                f"[{name}] SYNTAX ERROR at rendered line {e.lineno} "
                f"(Swift line {swift_line}): {e.msg}"
            )
            print(f"    >>> {e.text.strip() if e.text else ''}")
            failures += 1
    if checked == 0:
        print("error: no Python templates discovered")
        return 1
    print(
        f"checked {checked} embedded Python templates: "
        f"{'FAIL' if failures else 'all parse OK'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
