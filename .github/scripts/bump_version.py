import re
import sys
from pathlib import Path

def bump_version_string(version_str: str) -> str:
    parts = version_str.strip().split('.')
    if len(parts) >= 3 and parts[-1].isdigit():
        parts[-1] = str(int(parts[-1]) + 1)
        return '.'.join(parts)
    elif len(parts) == 2 and parts[-1].isdigit():
        parts[-1] = str(int(parts[-1]) + 1)
        return '.'.join(parts)
    else:
        # Fallback if non-standard
        return f"{version_str}.1"

def update_file(file_path: Path, pattern: str, replacement_fn):
    content = file_path.read_text(encoding="utf-8")
    new_content, count = re.subn(pattern, replacement_fn, content, count=1)
    if count > 0:
        file_path.write_text(new_content, encoding="utf-8")
        print(f"Updated {file_path.name}")
    else:
        print(f"Warning: Pattern not found in {file_path.name}")

def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    pyproject_path = root_dir / "pyproject.toml"
    setup_path = root_dir / "setup.py"
    init_path = root_dir / "pymbbo" / "__init__.py"

    # Read current version from pyproject.toml
    pyproject_content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'version\s*=\s*"([^"]+)"', pyproject_content)
    if not match:
        print("Error: Could not find version in pyproject.toml")
        sys.exit(1)

    current_version = match.group(1)
    new_version = bump_version_string(current_version)
    print(f"Bumping version: {current_version} -> {new_version}")

    # 1. Update pyproject.toml
    update_file(
        pyproject_path,
        r'(version\s*=\s*")[^"]+(")',
        lambda m: f'{m.group(1)}{new_version}{m.group(2)}'
    )

    # 2. Update setup.py
    if setup_path.exists():
        update_file(
            setup_path,
            r'(version\s*=\s*")[^"]+(")',
            lambda m: f'{m.group(1)}{new_version}{m.group(2)}'
        )

    # 3. Update pymbbo/__init__.py
    if init_path.exists():
        update_file(
            init_path,
            r'(__version__\s*=\s*")[^"]+(")',
            lambda m: f'{m.group(1)}{new_version}{m.group(2)}'
        )

    print(f"::set-output name=new_version::{new_version}")
    # Also write to GITHUB_OUTPUT if available
    import os
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"new_version={new_version}\n")

if __name__ == "__main__":
    main()
