from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def resolve_dataset_root() -> Path:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        raise FileNotFoundError("/kaggle/input does not exist")

    preferred = input_root / "industrial-heat-drone-sim-artifacts"
    if preferred.exists():
        return preferred

    # Fallback: detect any mounted dataset that contains the training payload.
    candidates = [
        p for p in input_root.iterdir()
        if p.is_dir() and (
            (p / "train_dyna_q.py").exists()
            or (p / "requirements-kaggle.txt").exists()
            or (p / "dyna_q_policy.json").exists()
        )
    ]
    if not candidates:
        available = sorted(p.name for p in input_root.iterdir() if p.is_dir())
        raise FileNotFoundError(
            "Could not find mounted dataset under /kaggle/input. "
            f"Available mounts: {available}"
        )

    if len(candidates) > 1:
        print("Multiple dataset candidates found, using first:")
        for c in candidates:
            print(f"  - {c}")

    return candidates[0]


def resolve_project_path(dataset_root: Path) -> Path | None:
    project_id = os.getenv("PROJECT_ID", "").strip()
    if not project_id:
        active_id_file = dataset_root / "active_project_id.txt"
        if active_id_file.exists():
            project_id = active_id_file.read_text(encoding="utf-8").strip()
            if project_id:
                print(f"Using active project id from file: {project_id}")
    projects_root = dataset_root / "projects"
    if not projects_root.exists():
        return None

    if project_id:
        selected = projects_root / project_id
        if selected.exists():
            return selected
        print(f"PROJECT_ID provided but not found: {selected}")

    candidates = [p for p in projects_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    print(f"No explicit PROJECT_ID found, using: {candidates[0].name}")
    return candidates[0]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    started_at = time.time()
    dataset_root = resolve_dataset_root()
    print(f"Using dataset root: {dataset_root}")
    project_path = resolve_project_path(dataset_root)
    if project_path:
        print(f"Using project payload: {project_path}")

    work_root = Path("/kaggle/working/project")
    trainer_root = work_root / "trainer"
    models_dir = trainer_root / "models"
    shared_dir = work_root / "shared"

    models_dir.mkdir(parents=True, exist_ok=True)
    shared_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(dataset_root / "train_dyna_q.py", trainer_root / "train_dyna_q.py")
    shutil.copy2(dataset_root / "requirements-kaggle.txt", trainer_root / "requirements-kaggle.txt")

    if (dataset_root / "sim_experience.json").exists():
        shutil.copy2(dataset_root / "sim_experience.json", shared_dir / "sim_experience.json")

    if (dataset_root / "dyna_q_policy.json").exists():
        shutil.copy2(dataset_root / "dyna_q_policy.json", models_dir / "dyna_q_policy.json")

    if project_path:
        # Prefer project-specific warm-start artifacts when available.
        if (project_path / "sim_experience.json").exists():
            shutil.copy2(project_path / "sim_experience.json", shared_dir / "sim_experience.json")
        if (project_path / "dyna_q_policy.json").exists():
            shutil.copy2(project_path / "dyna_q_policy.json", models_dir / "dyna_q_policy.json")

    run(["python", "-m", "pip", "install", "-r", str(trainer_root / "requirements-kaggle.txt")])

    train_seconds = os.getenv("DYNA_TRAIN_SECONDS", "5400")
    planning_steps = os.getenv("DYNA_PLANNING_STEPS", "60")
    episodes = os.getenv("DYNA_EPISODES", "15000")

    env = os.environ.copy()
    env["DYNA_TRAIN_SECONDS"] = train_seconds
    env["DYNA_PLANNING_STEPS"] = planning_steps
    env["DYNA_EPISODES"] = episodes

    print(
        f"Training with DYNA_TRAIN_SECONDS={train_seconds}, "
        f"DYNA_PLANNING_STEPS={planning_steps}, DYNA_EPISODES={episodes}"
    )

    subprocess.run(["python", "train_dyna_q.py"], cwd=str(trainer_root), env=env, check=True)

    output_root = Path("/kaggle/working/output")
    output_root.mkdir(parents=True, exist_ok=True)

    trained_policy = models_dir / "dyna_q_policy.json"
    if trained_policy.exists():
        shutil.copy2(trained_policy, output_root / "dyna_q_policy.json")
        print(f"Saved output policy: {output_root / 'dyna_q_policy.json'}")
    else:
        raise FileNotFoundError("Training finished but policy file was not generated.")

    elapsed = round(time.time() - started_at, 2)
    report = {
        "status": "ok",
        "dataset_root": str(dataset_root),
        "project_id": project_path.name if project_path else None,
        "training": {
            "dyna_train_seconds": train_seconds,
            "dyna_planning_steps": planning_steps,
            "dyna_episodes": episodes,
            "elapsed_seconds": elapsed,
        },
        "artifacts": {
            "policy": "dyna_q_policy.json",
        },
    }
    write_json(output_root / "training_report.json", report)
    print(f"Saved output report: {output_root / 'training_report.json'}")


if __name__ == "__main__":
    main()
