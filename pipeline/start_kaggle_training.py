from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run(cmd: list[str], env: dict | None = None) -> int:
    print("[run]", " ".join(cmd))
    completed = subprocess.run(cmd, env=env)
    return completed.returncode


def backup_current_policy(repo_root: Path) -> Path | None:
    policy = repo_root / 'trainer' / 'models' / 'dyna_q_policy.json'
    if not policy.exists():
        print('[backup] No existing policy to backup')
        return None
    backups = policy.parent / 'backups'
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    dst = backups / f'dyna_q_policy.backup.{stamp}.json'
    shutil.copy2(policy, dst)
    print(f'[backup] Copied current policy to: {dst}')
    return dst


def resolve_default_token(repo_root: Path) -> str | None:
    # Project-level default token file (optional)
    token_file = repo_root / 'pipeline' / 'default_kaggle_token.txt'
    if token_file.exists():
        return token_file.read_text(encoding='utf-8').strip()
    # Or from environment
    return os.getenv('DEFAULT_KAGGLE_API_TOKEN')


def main() -> None:
    parser = argparse.ArgumentParser(description='Start Kaggle training flow (wrapper around full_automation)')
    parser.add_argument('--repo-root', default='.', help='Repository root')
    parser.add_argument('--map-image', required=True, help='Path to map image')
    parser.add_argument('--project-id', required=True, help='Project id to register')
    parser.add_argument('--client-name', default='cli-client')
    parser.add_argument('--use-default-key', action='store_true', help='Use repository default Kaggle API key')
    parser.add_argument('--kaggle-token', default='', help='Kaggle API token (overrides default when provided)')
    parser.add_argument('--dyna-train-seconds', default='5400')
    parser.add_argument('--dyna-planning-steps', default='60')
    parser.add_argument('--dyna-episodes', default='15000')
    parser.add_argument('--wait-kernel', action='store_true')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    # Backup current policy
    backup_current_policy(repo_root)

    # Resolve token
    token = ''
    if args.kaggle_token:
        token = args.kaggle_token
    elif args.use_default_key:
        token = resolve_default_token(repo_root) or ''

    if not token:
        print('[error] No Kaggle token provided. Use --kaggle-token or --use-default-key with a configured default token.')
        sys.exit(2)

    # Build command to invoke full_automation.py
    full_auto = Path(repo_root) / 'pipeline' / 'full_automation.py'
    python_exe = sys.executable
    cmd = [python_exe, str(full_auto),
           '--repo-root', str(repo_root),
           '--map-image', str(args.map_image),
           '--project-id', args.project_id,
           '--client-name', args.client_name,
           '--kaggle-token', token]

    # Pass wait flag
    if args.wait_kernel:
        cmd.append('--wait-kernel')

    # Set RL params as environment variables for local steps
    env = os.environ.copy()
    env['DYNA_TRAIN_SECONDS'] = str(args.dyna_train_seconds)
    env['DYNA_PLANNING_STEPS'] = str(args.dyna_planning_steps)
    env['DYNA_EPISODES'] = str(args.dyna_episodes)

    print('[info] Starting full automation with RL parameters:')
    print(f"  DYNA_TRAIN_SECONDS={env['DYNA_TRAIN_SECONDS']}")
    print(f"  DYNA_PLANNING_STEPS={env['DYNA_PLANNING_STEPS']}")
    print(f"  DYNA_EPISODES={env['DYNA_EPISODES']}")

    rc = run(cmd, env=env)
    if rc != 0:
        print(f'[error] full_automation.py exited with code {rc}')
        sys.exit(rc)

    print('[done] Kaggle training flow invoked. If --wait-kernel was set the artifacts should be promoted into shared/projects/<project>/training')


if __name__ == '__main__':
    main()
