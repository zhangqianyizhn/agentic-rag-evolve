"""Create a clean, detached control checkout without changing the main repository."""

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    repo, output, manifest = (p.resolve() for p in (args.repo_root, args.output, args.manifest))
    if output == repo or repo in output.parents or output.exists() or manifest.exists():
        raise ValueError("control checkout must be a new path outside the repository")
    if manifest == output or output in manifest.parents:
        raise ValueError("control manifest must be outside its checkout")
    revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify",
                               f"{args.revision}^{{commit}}"], check=True,
                              capture_output=True, text=True).stdout.strip()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", str(output), revision],
                   check=True, capture_output=True)
    result = {"schema_version": "deepread-control-checkout-v1",
              "source_root": str(output), "commit": revision}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
