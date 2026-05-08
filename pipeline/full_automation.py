from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple


def run(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    printable = " ".join(cmd)
    print(f"[run] {printable}")
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.stderr:
        print(completed.stderr.strip())
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {printable}")
    return completed


def read_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def slugify(value: str) -> str:
    slug = re.sub(r'[^a-z0-9-]+', '-', value.strip().lower())
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug or 'project'


def resolve_python_exe(repo_root: Path) -> str:
    venv_exe = repo_root / '.venv' / 'Scripts' / 'python.exe'
    if venv_exe.exists():
        return str(venv_exe)
    return sys.executable


def kaggle_cmd(repo_root: Path, *args: str) -> list[str]:
    return [resolve_python_exe(repo_root), '-m', 'kaggle.cli', *args]


def ensure_dataset_base_files(repo_root: Path) -> None:
    dataset_root = repo_root / 'kaggle' / 'dataset'
    trainer_root = repo_root / 'trainer'

    shutil.copy2(trainer_root / 'train_dyna_q.py', dataset_root / 'train_dyna_q.py')

    default_policy = trainer_root / 'models' / 'dyna_q_policy.json'
    if default_policy.exists():
        shutil.copy2(default_policy, dataset_root / 'dyna_q_policy.json')

    shared_exp = repo_root / 'shared' / 'sim_experience.json'
    if shared_exp.exists():
        shutil.copy2(shared_exp, dataset_root / 'sim_experience.json')


def build_manifest(
    manifest_path: Path,
    project_id: str,
    map_file_name: str,
    grid_resolution: int,
    world_scale: float,
    client_name: str,
) -> None:
    payload = {
        'project_id': slugify(project_id),
        'client_name': client_name,
        'map_image_file': map_file_name,
        'grid_resolution': grid_resolution,
        'world_scale_m_per_cell': world_scale,
        'mission_profile': {
            'max_seconds': 420,
            'drone_max_speed': 18,
            'heat_zone_budget': 36,
        },
    }
    write_json(manifest_path, payload)


def prepare_project(repo_root: Path, map_image: Path, manifest_path: Path) -> str:
    script = repo_root / 'trainer' / 'cv_pipeline' / 'project_orchestrator.py'
    python_exe = repo_root / '.venv' / 'Scripts' / 'python.exe'
    python_cmd = str(python_exe) if python_exe.exists() else sys.executable

    if not script.exists():
        manifest = read_json(manifest_path)
        project_id = manifest['project_id']
        shared_root = repo_root / 'shared' / 'projects' / project_id
        raw_root = shared_root / 'raw'
        processed_root = shared_root / 'processed'
        raw_root.mkdir(parents=True, exist_ok=True)
        processed_root.mkdir(parents=True, exist_ok=True)

        shutil.copy2(map_image, raw_root / map_image.name)
        shutil.copy2(map_image, processed_root / map_image.name)
        shutil.copy2(manifest_path, raw_root / 'project_manifest.json')
        shutil.copy2(manifest_path, processed_root / 'project_manifest.json')

        env_spec_path = os.getenv('TRAINING_ENVIRONMENT_SPEC_PATH', '')
        env_spec = safe_read_json(Path(env_spec_path)) if env_spec_path else None
        if env_spec is None:
          env_spec = {
              'project_id': project_id,
              'source': 'auto-generated',
              'grid': {'width': 40, 'height': 40, 'cell_size': 3},
              'rendering': {
                  'wall_height': 18,
                  'floor_material': 'concrete',
                  'wall_material': 'steel',
                  'lighting_profile': 'industrial-default',
                  'fog_density': 0.004,
              },
              'optimization': {
                  'merge_static_geometry': True,
                  'use_instancing': True,
                  'enable_lod': True,
                  'target_fps': 60,
                  'max_draw_calls': 220,
              },
          }
        env_spec['project_id'] = project_id
        write_json(processed_root / 'env_3d_spec.json', env_spec)

        objects = env_spec.get('objects', []) if isinstance(env_spec, dict) else []
        occupancy = {
            'project_id': project_id,
            'source': 'fallback-generator',
            'map_image': map_image.name,
            'objects': objects,
            'grid_resolution': manifest.get('grid_resolution', 80),
            'world_scale_m_per_cell': manifest.get('world_scale_m_per_cell', 1.4),
        }
        write_json(processed_root / 'occupancy_grid.json', occupancy)

        camera_presets = {
            'project_id': project_id,
            'presets': [
                {'name': 'topdown', 'position': {'x': 0, 'y': 220, 'z': 0}, 'target': {'x': 0, 'y': 0, 'z': 0}},
                {'name': 'follow', 'position': {'x': 24, 'y': 26, 'z': 24}, 'target': {'x': 0, 'y': 0, 'z': 0}},
                {'name': 'isometric', 'position': {'x': 120, 'y': 90, 'z': 120}, 'target': {'x': 0, 'y': 0, 'z': 0}},
            ],
        }
        write_json(processed_root / 'camera_presets.json', camera_presets)

        lighting_profile = {
            'project_id': project_id,
            'profile': env_spec.get('rendering', {}).get('lighting_profile', 'industrial-default') if isinstance(env_spec, dict) else 'industrial-default',
            'fog_density': env_spec.get('rendering', {}).get('fog_density', 0.004) if isinstance(env_spec, dict) else 0.004,
        }
        write_json(processed_root / 'lighting_profile.json', lighting_profile)

        return project_id

    run(
        [
            python_cmd,
            str(script),
            '--project-root',
            str(repo_root),
            '--map-image',
            str(map_image),
            '--manifest',
            str(manifest_path),
            '--seed-policy',
            str(repo_root / 'trainer' / 'models' / 'dyna_q_policy.json'),
            '--sim-experience',
            str(repo_root / 'shared' / 'sim_experience.json'),
        ]
    )

    manifest = read_json(manifest_path)
    return manifest['project_id']


def register_project(repo_root: Path, project_id: str) -> None:
    script = repo_root / 'pipeline' / 'register_kaggle_project.py'
    python_exe = repo_root / '.venv' / 'Scripts' / 'python.exe'
    python_cmd = str(python_exe) if python_exe.exists() else sys.executable
    run([
        python_cmd,
        str(script),
        '--repo-root',
        str(repo_root),
        '--project-id',
        project_id,
    ])


def set_active_project(repo_root: Path, project_id: str) -> None:
    active_file = repo_root / 'kaggle' / 'dataset' / 'active_project_id.txt'
    active_file.write_text(project_id, encoding='utf-8')


def publish_dataset(repo_root: Path, token: str, message: str) -> None:
    env = os.environ.copy()
    env['KAGGLE_API_TOKEN'] = token
    dataset_dir = repo_root / 'kaggle' / 'dataset'

    version_cmd = kaggle_cmd(repo_root, 'datasets', 'version', '-p', str(dataset_dir), '--dir-mode', 'zip', '-m', message)
    result = run(version_cmd, env=env, check=False)
    if result.returncode == 0:
        return

    if 'not found' in (result.stdout + result.stderr).lower() or 'does not exist' in (result.stdout + result.stderr).lower():
        create_cmd = kaggle_cmd(repo_root, 'datasets', 'create', '-p', str(dataset_dir), '--dir-mode', 'zip')
        run(create_cmd, env=env)
        return

    raise RuntimeError(f'Failed to publish dataset version to Kaggle: {result.stdout}\n{result.stderr}')


def push_kernel(repo_root: Path, token: str) -> None:
    env = os.environ.copy()
    env['KAGGLE_API_TOKEN'] = token
    kernel_dir = repo_root / 'kaggle' / 'kernel'
    run(kaggle_cmd(repo_root, 'kernels', 'push', '-p', str(kernel_dir)), env=env)


def poll_kernel(
    repo_root: Path,
    token: str,
    kernel_ref: str,
    timeout_minutes: int,
    poll_seconds: int,
) -> str:
    env = os.environ.copy()
    env['KAGGLE_API_TOKEN'] = token

    deadline = time.time() + timeout_minutes * 60
    latest = ''
    while time.time() < deadline:
        result = run(kaggle_cmd(repo_root, 'kernels', 'status', kernel_ref), env=env)
        text = (result.stdout + '\n' + result.stderr).strip()
        latest = text
        if 'KernelWorkerStatus.COMPLETE' in text:
            return 'complete'
        if 'KernelWorkerStatus.ERROR' in text:
            return 'error'
        print(f"[wait] Kernel still running. Next check in {poll_seconds}s")
        time.sleep(poll_seconds)

    print(latest)
    return 'timeout'


def download_outputs(repo_root: Path, token: str, kernel_ref: str, project_id: str) -> Path:
    env = os.environ.copy()
    env['KAGGLE_API_TOKEN'] = token

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = repo_root / 'kaggle' / 'kernel' / 'outputs' / project_id / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    run(kaggle_cmd(repo_root, 'kernels', 'output', kernel_ref, '-p', str(out_dir)), env=env)
    return out_dir


def promote_artifacts(repo_root: Path, project_id: str, output_dir: Path) -> Tuple[Path | None, Path | None]:
    policy_src = output_dir / 'dyna_q_policy.json'
    report_src = output_dir / 'training_report.json'

    project_training = repo_root / 'shared' / 'projects' / project_id / 'training'
    project_delivery = repo_root / 'shared' / 'projects' / project_id / 'delivery'
    project_training.mkdir(parents=True, exist_ok=True)
    project_delivery.mkdir(parents=True, exist_ok=True)

    policy_dst = None
    report_dst = None

    if policy_src.exists():
        policy_dst = project_training / 'dyna_q_policy.json'
        shutil.copy2(policy_src, policy_dst)
        shutil.copy2(policy_src, repo_root / 'trainer' / 'models' / 'dyna_q_policy.json')

    if report_src.exists():
        report_dst = project_training / 'training_report.json'
        shutil.copy2(report_src, report_dst)

    delivery_manifest = {
        'project_id': project_id,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'artifacts': {
            'occupancy_grid': '../processed/occupancy_grid.json',
            'semantic_map': '../processed/semantic_map.json',
            'env_3d_spec': '../processed/env_3d_spec.json',
            'camera_presets': '../processed/camera_presets.json',
            'lighting_profile': '../processed/lighting_profile.json',
            'policy': '../training/dyna_q_policy.json',
            'training_report': '../training/training_report.json',
        },
        'simulation': {
            'mode': 'policy-driven',
            'fallback_mode': 'explore-only',
        },
    }
    write_json(project_delivery / 'simulation_summary.json', delivery_manifest)

    # Publish a web-loadable project bundle for immediate simulation playback.
    web_bundle = repo_root / 'web' / 'public' / 'projects' / project_id
    web_bundle.mkdir(parents=True, exist_ok=True)

    project_processed = repo_root / 'shared' / 'projects' / project_id / 'processed'
    copy_pairs = [
        (project_processed / 'occupancy_grid.json', web_bundle / 'occupancy_grid.json'),
        (project_processed / 'semantic_map.json', web_bundle / 'semantic_map.json'),
        (project_processed / 'env_3d_spec.json', web_bundle / 'env_3d_spec.json'),
        (project_processed / 'camera_presets.json', web_bundle / 'camera_presets.json'),
        (project_processed / 'lighting_profile.json', web_bundle / 'lighting_profile.json'),
        (project_training / 'dyna_q_policy.json', web_bundle / 'dyna_q_policy.json'),
        (project_training / 'training_report.json', web_bundle / 'training_report.json'),
    ]
    for src, dst in copy_pairs:
        if src.exists():
            shutil.copy2(src, dst)

    web_manifest = {
        'project_id': project_id,
        'base_path': f'/projects/{project_id}',
        'files': {
            'occupancy_grid': 'occupancy_grid.json',
            'semantic_map': 'semantic_map.json',
            'env_3d_spec': 'env_3d_spec.json',
            'camera_presets': 'camera_presets.json',
            'lighting_profile': 'lighting_profile.json',
            'policy': 'dyna_q_policy.json',
            'training_report': 'training_report.json',
        },
    }
    write_json(web_bundle / 'project_manifest.json', web_manifest)

    return policy_dst, report_dst


def main() -> None:
    parser = argparse.ArgumentParser(description='Full end-to-end automation: upload map -> Kaggle train -> simulation package')
    parser.add_argument('--repo-root', required=True)
    parser.add_argument('--map-image', required=True)
    parser.add_argument('--project-id', required=True)
    parser.add_argument('--client-name', default='unknown-client')
    parser.add_argument('--grid-resolution', type=int, default=80)
    parser.add_argument('--world-scale', type=float, default=1.4)
    parser.add_argument('--kaggle-token', default='')
    parser.add_argument('--kaggle-owner', default='anshubhawsar')
    parser.add_argument('--dataset-slug', default='industrial-heat-drone-sim-artifacts')
    parser.add_argument('--kernel-slug', default='industrial-heat-drone-gpu-trainer')
    parser.add_argument('--wait-kernel', action='store_true')
    parser.add_argument('--timeout-minutes', type=int, default=180)
    parser.add_argument('--poll-seconds', type=int, default=45)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    map_image = Path(args.map_image).resolve()
    project_id = slugify(args.project_id)

    if not map_image.exists():
        raise FileNotFoundError(f'Map image not found: {map_image}')

    pipeline_dir = repo_root / 'pipeline' / 'generated_manifests'
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pipeline_dir / f'{project_id}.manifest.json'
    build_manifest(
        manifest_path=manifest_path,
        project_id=project_id,
        map_file_name=map_image.name,
        grid_resolution=args.grid_resolution,
        world_scale=args.world_scale,
        client_name=args.client_name,
    )

    print('[phase] Prepare project artifacts')
    project_id = prepare_project(repo_root=repo_root, map_image=map_image, manifest_path=manifest_path)

    print('[phase] Register project in Kaggle dataset structure')
    register_project(repo_root=repo_root, project_id=project_id)

    print('[phase] Refresh dataset base files and active project marker')
    ensure_dataset_base_files(repo_root)
    set_active_project(repo_root=repo_root, project_id=project_id)

    kaggle_token = args.kaggle_token or os.getenv('KAGGLE_API_TOKEN', '')
    if not kaggle_token:
        raise RuntimeError('Kaggle token missing. Pass --kaggle-token or set KAGGLE_API_TOKEN')

    message = f'auto-flow update for {project_id}'

    print('[phase] Publish Kaggle dataset and trigger kernel')
    publish_dataset(repo_root=repo_root, token=kaggle_token, message=message)
    push_kernel(repo_root=repo_root, token=kaggle_token)

    kernel_ref = f"{args.kaggle_owner}/{args.kernel_slug}"
    if not args.wait_kernel:
        print('[done] Kernel triggered. Use --wait-kernel to block until completion and auto-download outputs.')
        return

    print('[phase] Wait for kernel completion')
    status = poll_kernel(
        repo_root=repo_root,
        token=kaggle_token,
        kernel_ref=kernel_ref,
        timeout_minutes=args.timeout_minutes,
        poll_seconds=args.poll_seconds,
    )
    if status != 'complete':
        raise RuntimeError(f'Kernel did not complete successfully. Final status: {status}')

    print('[phase] Download outputs and promote to simulation package')
    out_dir = download_outputs(repo_root=repo_root, token=kaggle_token, kernel_ref=kernel_ref, project_id=project_id)
    policy_dst, report_dst = promote_artifacts(repo_root=repo_root, project_id=project_id, output_dir=out_dir)

    print('[success] Full automation complete')
    print(f'Project: {project_id}')
    print(f'Kaggle output folder: {out_dir}')
    if policy_dst:
        print(f'Policy promoted to: {policy_dst}')
    if report_dst:
        print(f'Report promoted to: {report_dst}')


if __name__ == '__main__':
    main()
