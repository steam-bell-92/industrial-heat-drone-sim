# Standard Project Directory Contract

Every user map project must follow this structure.

```text
shared/projects/<project_id>/
  raw/
    map_image.png
    project_manifest.json
  processed/
    occupancy_grid.json
    occupancy_preview.png
    semantic_map.json
    env_3d_spec.json
    camera_presets.json
    lighting_profile.json
  training/
    dyna_q_policy.json
    training_report.json
  delivery/
    simulation_summary.json
    mission_report.pdf

kaggle/dataset/projects/<project_id>/
  project_manifest.json
  occupancy_grid.json
  env_3d_spec.json
  sim_experience.json          # optional
  dyna_q_policy.json           # optional warm start
```

## Naming rules

- `project_id` must be lowercase slug: `[a-z0-9-]+`
- Keep all map-derived files under `processed/`
- Keep all model artifacts under `training/`
- Never overwrite historical policy without version suffix

## Recommended versioning

- `dyna_q_policy.v1.json`
- `dyna_q_policy.v2.json`
- `training_report.2026-04-16T21-05-00Z.json`
