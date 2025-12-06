"""Generate the example pages and navigation."""

import ast
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
examples_dir = root / "examples"

# Categories to include (subdirectories of examples/)
CATEGORIES = ["abstract", "car", "drone", "realtime"]

# Files to skip (utility modules, not examples)
SKIP_FILES = {"plotting.py", "__init__.py"}


def get_module_docstring(file_path: Path) -> str | None:
    """Extract the module-level docstring from a Python file."""
    try:
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())
        return ast.get_docstring(tree)
    except Exception:
        return None


def format_title(name: str) -> str:
    """Convert a file name to a human-readable title."""
    # Remove .py extension and convert underscores/hyphens to spaces
    title = name.replace("_", " ").replace("-", " ")
    # Capitalize words, handling special cases
    words = title.split()
    formatted_words = []
    for word in words:
        # Keep acronyms and numbers as-is
        if word.isupper() or word[0].isdigit():
            formatted_words.append(word)
        else:
            formatted_words.append(word.capitalize())
    return " ".join(formatted_words)


for category in CATEGORIES:
    category_path = examples_dir / category
    if not category_path.exists():
        continue

    for path in sorted(category_path.glob("*.py")):
        if path.name in SKIP_FILES or path.name.startswith("_"):
            continue

        # Create the documentation path
        rel_path = path.relative_to(examples_dir)
        doc_path = rel_path.with_suffix(".md")
        full_doc_path = Path("examples", doc_path)

        # Get module docstring if present
        docstring = get_module_docstring(path)

        # Format the title
        title = format_title(path.stem)
        category_display = category.capitalize()

        # Build navigation entry
        nav_parts = (category_display, title)
        nav[nav_parts] = doc_path.as_posix()

        # Read the source code
        source_code = path.read_text()

        # Generate the markdown content
        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            fd.write(f"# {title}\n\n")

            if docstring:
                fd.write(f"{docstring}\n\n")

            fd.write("## Source Code\n\n")
            fd.write(f"**File:** `examples/{rel_path}`\n\n")
            fd.write("```python\n")
            fd.write(source_code)
            fd.write("\n```\n")

        # Set edit path to the original Python file
        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# Write the navigation file for literate-nav
with mkdocs_gen_files.open("examples/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
