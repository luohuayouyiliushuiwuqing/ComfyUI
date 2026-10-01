import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "models"
OUTPUT = Path(__file__).resolve().parent / "models_tree.txt"


def build_tree(directory: Path, prefix: str = ""):
    entries = sorted(
        directory.iterdir(),
        key=lambda p: (p.is_file(), p.name.lower()),
    )
    lines = []
    for i, entry in enumerate(entries):
        last = i == len(entries) - 1
        connector = "└── " if last else "├── "
        lines.append(prefix + connector + entry.name)
        if entry.is_dir():
            extension = "    " if last else "│   "
            lines.extend(build_tree(entry, prefix + extension))
    return lines


def main():
    if not ROOT.is_dir():
        raise SystemExit(f"models directory not found: {ROOT}")

    lines = [str(ROOT), ""]
    lines.extend(build_tree(ROOT))

    text = "\n".join(lines) + "\n"
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
