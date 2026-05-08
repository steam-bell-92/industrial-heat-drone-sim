# Digital Twin Program Phases (2D -> 3D -> RL -> Delivery)

## Phase 1: Client Intake and Data Contract

Goal: Collect user map and required metadata in a strict, machine-readable format.

Inputs:
- `map_image` (PNG/JPG/BMP/TIFF)
- `project_manifest.json`
- optional `asset_preferences.json`

Deliverables:
- Validated manifest with project id and map scale
- Ingestion folder created under `shared/projects/<project_id>/raw`

Exit criteria:
- Map resolution is known
- Scale is known or defaulted
- No schema validation errors

## Phase 2: Computer Vision and 2D Semantic Extraction

Goal: Convert map image into occupancy and semantic layers.

Core steps:
1. Normalize image (resize, denoise, contrast normalization)
2. Detect walls and hard obstacles (adaptive threshold + morphology)
3. Generate occupancy matrix (`0=free`, `1=blocked`)
4. Generate semantic layer (`road`, `building`, `restricted`, `rail`, `utility`)

Deliverables:
- `occupancy_grid.json`
- `occupancy_preview.png`
- `semantic_map.json`

Exit criteria:
- Occupancy ratio is in a safe range
- Connected free-space component exists

## Phase 3: 3D Environment Synthesis (Mind-Blowing but Optimal)

Goal: Build a high-fidelity but efficient 3D environment from semantic + occupancy data.

Design principles:
- Instanced meshes for repeated objects
- Merged geometries for static structures
- LOD tiers for distant detail
- Material presets by semantic class
- Procedural props with deterministic seed

Deliverables:
- `env_3d_spec.json`
- `camera_presets.json`
- `lighting_profile.json`

Exit criteria:
- Scene can render at target fps
- Geometry count and draw calls under thresholds

## Phase 4: RL Training (Kaggle GPU)

Goal: Train map-aware policy on client environment constraints.

Core steps:
1. Package project artifacts into Kaggle dataset payload
2. Push dataset version
3. Run GPU kernel (`train_on_kaggle.py`)
4. Export policy and training report

Deliverables:
- `dyna_q_policy.json` (or PPO policy in future)
- `training_report.json`
- `kaggle_run_manifest.json`

Exit criteria:
- Policy generated successfully
- Success metrics meet baseline threshold

## Phase 5: Simulation Playback and Delivery

Goal: Show client drone behavior in their own environment.

Core steps:
1. Load `env_3d_spec.json` in web simulator
2. Load policy artifact
3. Run mission playback and KPI overlays
4. Export mission report PDF + JSON

Deliverables:
- Policy package
- Simulation playback demo
- KPI summary (`coverage`, `time_to_detection`, `safety`)

Exit criteria:
- Client can replay mission and inspect telemetry
- All artifacts versioned and reproducible

## Phase 6: Continuous Improvement Loop

Goal: Improve accuracy and realism over multiple dataset versions.

- Add failed trajectories to experience buffer
- Version maps and policies per client project
- Schedule retraining cycles on Kaggle GPU
- Promote only metrics-approved policies to production
