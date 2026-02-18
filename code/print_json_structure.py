#!/usr/bin/env python3
"""
Print a concise tree view of a JSON document's structure (types only, not values).

Features:
- Distinguish between object, array, string, integer, number, boolean, null
- Sample only the first N items of arrays to infer structure
- Optional max depth
- Optional key sorting and array length display
- Works from file path or STDIN

Example:
    python /home/code/3Ddetection/IS-Fusion/GenKG/code/print_json_structure.py -f /home/code/3Ddetection/IS-Fusion/GenKG/data/sorted_by_scene_ISFUSIONandGTattr.json  
    cat sample.json | python json_tree.py
    python json_tree.py -f sample.json -n 3 -d 5 --no-sort
"""

import sys
import json
import argparse
from collections.abc import Mapping, Sequence

PRIMITIVES = (str, int, float, bool, type(None))

def type_name(v):
    """Return a human-readable JSON type name for a Python value."""
    if v is None:
        return "null"
    # bool must be checked before int (bool is a subclass of int in Python)
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int) and not isinstance(v, bool):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, Mapping):
        return "object"
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
        return "array"
    return type(v).__name__

def print_json_structure(
    data,
    name="root",
    indent=0,
    *,
    sample_list=1,
    max_depth=None,
    show_lengths=True,
    sort_keys=True,
):
    """
    Recursively print a JSON structure tree (no concrete values).

    Args:
        data: Parsed JSON (dict/list/primitive).
        name: Label for the current node in the tree.
        indent: Current indentation level.
        sample_list: For arrays, sample the first N items to infer structure.
        max_depth: Stop descending once this depth is reached (None = unlimited).
        show_lengths: If True, show array length when available.
        sort_keys: If True, sort object keys alphabetically.
    """
    prefix = "  " * indent
    tn = type_name(data)

    extra = ""
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)) and show_lengths:
        try:
            extra = f" len={len(data)}"
        except Exception:
            pass

    print(f"{prefix}{name}: {tn}{extra}")

    # Respect depth limit
    if max_depth is not None and indent >= max_depth:
        return

    # Objects (mappings)
    if isinstance(data, Mapping):
        keys = data.keys()
        if sort_keys:
            keys = sorted(keys)
        for k in keys:
            print_json_structure(
                data[k],
                name=str(k),
                indent=indent + 1,
                sample_list=sample_list,
                max_depth=max_depth,
                show_lengths=show_lengths,
                sort_keys=sort_keys,
            )

    # Arrays (sequences that are not strings/bytes)
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        if not data:
            print(f"{prefix}  (empty)")
            return
        n = min(max(1, int(sample_list)), len(data))
        sampled = data[:n]

        # If sampled items have mixed types, show a quick union summary
        kinds = {type_name(x) for x in sampled}
        if len(kinds) > 1:
            kinds_str = " | ".join(sorted(kinds))
            print(f"{prefix}  items: {kinds_str}")

        # Show structure for the first N items
        for i, item in enumerate(sampled):
            print_json_structure(
                item,
                name=f"[{i}]",
                indent=indent + 1,
                sample_list=sample_list,
                max_depth=max_depth,
                show_lengths=show_lengths,
                sort_keys=sort_keys,
            )

def _load_json_from_file(path: str):
    """Load JSON from a file path."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_json_from_stdin():
    """Load JSON from STDIN."""
    text = sys.stdin.read()
    if not text.strip():
        raise ValueError("No input detected on STDIN.")
    return json.loads(text)

def main():
    parser = argparse.ArgumentParser(
        description="Print a concise tree view of a JSON document's structure."
    )
    parser.add_argument(
        "-f", "--file",
        help="Path to a JSON file. If omitted, JSON is read from STDIN."
    )
    parser.add_argument(
        "-n", "--sample-list",
        type=int, default=1,
        help="Sample the first N array items (default: 1)."
    )
    parser.add_argument(
        "-d", "--max-depth",
        type=int, default=None,
        help="Limit recursion depth (default: unlimited)."
    )
    parser.add_argument(
        "--no-lengths",
        action="store_true",
        help="Do not display array lengths."
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Do not sort object keys (preserve original order)."
    )
    parser.add_argument(
        "-k", "--root-name",
        default="root",
        help='Label used for the top-level node (default: "root").'
    )

    args = parser.parse_args()

    # Determine input source
    try:
        if args.file:
            data = _load_json_from_file(args.file)
        else:
            # If no file is provided, expect input via STDIN
            if sys.stdin.isatty():
                parser.print_help(sys.stderr)
                sys.exit(2)
            data = _load_json_from_stdin()
    except Exception as e:
        print(f"Error loading JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Print the structure
    print_json_structure(
        data,
        name=args.root_name,
        sample_list=args.sample_list,
        max_depth=args.max_depth,
        show_lengths=not args.no_lengths,
        sort_keys=not args.no_sort,
    )

if __name__ == "__main__":
    main()
