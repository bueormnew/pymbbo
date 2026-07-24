import re
import subprocess
import sys
from pathlib import Path

def get_commit_count() -> int:
    try:
        res = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return int(res.stdout.strip())
    except Exception as e:
        print(f"Warning: Could not get commit count from git: {e}")
        return 1

def update_file(file_path: Path, pattern: str, replacement_fn):
    if not file_path.exists():
        return
    content = file_path.read_text(encoding="utf-8")
    new_content, count = re.subn(pattern, replacement_fn, content, count=1)
    if count > 0:
        file_path.write_text(new_content, encoding="utf-8")
        print(f"Updated {file_path.name}")

def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    pyproject_path = root_dir / "pyproject.toml"
    setup_path = root_dir / "setup.py"
    init_path = root_dir / "pymbbo" / "__init__.py"

    pyproject_content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'version\s*=\s*"([^"]+)"', pyproject_content)
    if not match:
        print("Error: Could not find version in pyproject.toml")
        sys.exit(1)

    base_version = match.group(1)
    parts = base_version.strip().split('.')
    major_minor = '.'.join(parts[:2]) if len(parts) >= 2 else "0.1"

    commit_count = get_commit_count()
    new_version = f"{major_minor}.{commit_count}"

    print(f"Generated PyPI version: {new_version}")

    # 1. Update pyproject.toml
    update_file(
        pyproject_path,
        r'(version\s*=\s*")[^"]+(")',
        lambda m: f'{m.group(1)}{new_version}{m.group(2)}'
    )

    # 2. Update setup.py
    update_file(
        setup_path,
        r'(version\s*=\s*")[^"]+(")',
        lambda m: f'{m.group(1)}{new_version}{m.group(2)}'
    )

    # 3. Update pymbbo/__init__.py
    update_file(
        init_path,
        r'(__version__\s*=\s*")[^"]+(")',
        lambda m: f'{m.group(1)}{new_version}{m.group(2)}'
    )

    import os
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"new_version={new_version}\n")

if __name__ == "__main__":
    main()
