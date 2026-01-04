from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


def default_custom_nodes_dir() -> Path:
    return Path(__file__).resolve().parent / "custom_nodes"


def derive_repo_name(repo_url: str) -> str:
    parsed = urlparse(repo_url)
    repo_name = Path(parsed.path).name
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    if not repo_name:
        raise ValueError(f"Could not determine repository name from url: {repo_url}")
    return repo_name


def build_clone_command(repo_url: str, destination: Path, branch: Optional[str]) -> list[str]:
    command = ["git", "clone"]
    if branch:
        command.extend(["--branch", branch])
    command.extend([repo_url, str(destination)])
    return command


def build_pull_command(destination: Path, branch: Optional[str]) -> list[str]:
    command = ["git", "-C", str(destination), "pull"]
    if branch:
        command.extend(["origin", branch])
    return command


def install_custom_node(
    repo_url: str,
    *,
    destination_dir: Path,
    repo_name: Optional[str],
    branch: Optional[str],
    dry_run: bool,
) -> Path:
    resolved_name = repo_name or derive_repo_name(repo_url)
    target_dir = destination_dir / resolved_name

    if dry_run:
        operation = "update" if target_dir.exists() else "clone"
        print(f"[dry-run] Would {operation} {repo_url} -> {target_dir}")
        return target_dir

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if target_dir.exists():
        if not (target_dir / ".git").exists():
            raise RuntimeError(f"Destination exists and is not a git repository: {target_dir}")
        command = build_pull_command(target_dir, branch)
    else:
        command = build_clone_command(repo_url, target_dir, branch)

    subprocess.run(command, check=True)
    return target_dir


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install or update a ComfyUI custom node from a git repository. "
            "By default the repository is cloned into the local 'custom_nodes' directory."
        )
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Git repository URL to install, e.g. https://github.com/Yveske77/Tune-Forge-1",
    )
    parser.add_argument(
        "--custom-nodes-dir",
        type=Path,
        default=default_custom_nodes_dir(),
        help="Target directory for custom nodes (default: ./custom_nodes)",
    )
    parser.add_argument(
        "--name",
        help="Override the folder name to clone into. Defaults to the repository name.",
    )
    parser.add_argument(
        "--branch",
        help="Optional branch name to clone or update.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without cloning or updating repositories.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        target = install_custom_node(
            args.repo,
            destination_dir=args.custom_nodes_dir,
            repo_name=args.name,
            branch=args.branch,
            dry_run=args.dry_run,
        )
    except (subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    action = "Planned installation into" if args.dry_run else "Installed/updated custom node at"
    print(f"{action} {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
