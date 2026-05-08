from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def register_project(repo_root: Path, project_id: str) -> Path:
    shared_root = repo_root / 'shared' / 'projects' / project_id
    processed = shared_root / 'processed'
    raw = shared_root / 'raw'

    if not shared_root.exists():
        raise FileNotFoundError(f'Project not found: {shared_root}')

    target = repo_root / 'kaggle' / 'dataset' / 'projects' / project_id
    target.mkdir(parents=True, exist_ok=True)

    copy_if_exists(raw / 'project_manifest.json', target / 'project_manifest.json')
    copy_if_exists(processed / 'occupancy_grid.json', target / 'occupancy_grid.json')
    copy_if_exists(processed / 'env_3d_spec.json', target / 'env_3d_spec.json')

    run_manifest = {
        'project_id': project_id,
        'dataset_project_path': f'projects/{project_id}',
        'required_files': [
            'project_manifest.json',
            'occupancy_grid.json',
            'env_3d_spec.json',
        ],
        'optional_files': ['sim_experience.json', 'dyna_q_policy.json'],
    }
    write_json(target / 'kaggle_run_manifest.json', run_manifest)

    index_path = repo_root / 'kaggle' / 'dataset' / 'kaggle_projects_index.json'
    index = read_json(index_path) if index_path.exists() else {'schema_version': 1, 'projects': []}

    projects = index.get('projects', [])
    projects = [p for p in projects if p.get('project_id') != project_id]
    projects.append(
        {
            'project_id': project_id,
            'dataset_path': f'projects/{project_id}',
            'has_occupancy': (target / 'occupancy_grid.json').exists(),
            'has_env_spec': (target / 'env_3d_spec.json').exists(),
        }
    )
    index['projects'] = sorted(projects, key=lambda p: p['project_id'])
    write_json(index_path, index)

    return target


def main() -> None:
    parser = argparse.ArgumentParser(description='Register shared project artifacts into Kaggle dataset structure')
    parser.add_argument('--repo-root', required=True)
    parser.add_argument('--project-id', required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    project_id = args.project_id.strip().lower()

    out = register_project(repo_root=repo_root, project_id=project_id)
    print(f'Registered project payload at: {out}')


if __name__ == '__main__':
    main()
