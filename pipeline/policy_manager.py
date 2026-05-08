from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from datetime import datetime


def list_available_policies(repo_root: Path) -> list[Path]:
    policies = []
    trainer_models = repo_root / 'trainer' / 'models'
    if trainer_models.exists():
        for p in trainer_models.iterdir():
            if p.name.startswith('dyna_q_policy') and p.suffix == '.json':
                policies.append(p)
    # Also check shared project training folders
    shared_projects = repo_root / 'shared' / 'projects'
    if shared_projects.exists():
        for proj in shared_projects.iterdir():
            t = proj / 'training' / 'dyna_q_policy.json'
            if t.exists():
                policies.append(t)
    return policies


def backup_and_activate(repo_root: Path, source: Path) -> Path:
    current = repo_root / 'trainer' / 'models' / 'dyna_q_policy.json'
    backups = current.parent / 'backups'
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    if current.exists():
        b = backups / f'dyna_q_policy.pre_switch.{stamp}.json'
        shutil.copy2(current, b)
        print(f'[backup] Current policy backed up to {b}')
    shutil.copy2(source, current)
    print(f'[activate] Activated policy from {source} -> {current}')
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description='List and switch policies')
    parser.add_argument('--repo-root', default='.', help='Repository root')
    parser.add_argument('--activate', help='Path to policy file to activate')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    if args.activate:
        src = Path(args.activate).resolve()
        if not src.exists():
            print(f'[error] Not found: {src}')
            return
        backup_and_activate(repo_root, src)
        return

    policies = list_available_policies(repo_root)
    if not policies:
        print('No policies found in trainer/models or shared/projects/*/training')
        return
    print('Available policies:')
    for p in policies:
        print(' -', p)


if __name__ == '__main__':
    main()
