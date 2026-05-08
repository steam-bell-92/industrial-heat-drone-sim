# Kaggle Dataset Payload Contract

This dataset is the authoritative training payload for Kaggle GPU runs.

## Root files

- `train_dyna_q.py`: Trainer entry
- `requirements-kaggle.txt`: Kaggle dependency lock
- `dyna_q_policy.json`: Optional warm-start policy
- `sim_experience.json`: Optional replay/warm experience

## Project folders

`projects/<project_id>/`
- `project_manifest.json`
- `occupancy_grid.json`
- `env_3d_spec.json`
- `kaggle_run_manifest.json`
- optional: `sim_experience.json`, `dyna_q_policy.json`

## Rules

- Keep one project directory per client map
- Avoid deleting old project folders; version by `project_id`
- Always update dataset version before kernel push
