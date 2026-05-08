# Kaggle GPU Operational Flow (Per Client Project)

## One-Command Full Automation

If your map image is ready and `KAGGLE_API_TOKEN` is set, run:

```powershell
pipeline\run_full_flow.bat "c:\path\to\uploaded_map.png" "client-project-id" "Client Name"
```

This command executes all phases automatically:
1. Build project manifest
2. Convert map to occupancy + semantic + 3D spec
3. Register project under `kaggle/dataset/projects/<project_id>`
4. Publish dataset version
5. Trigger Kaggle GPU kernel
6. Wait for completion
7. Download outputs
8. Promote policy/report to `shared/projects/<project_id>/training`
9. Write simulation package in `shared/projects/<project_id>/delivery`
10. Publish web bundle in `web/public/projects/<project_id>`

Equivalent Python entrypoint:

```powershell
& ".venv/Scripts/python.exe" "pipeline/full_automation.py" \
  --repo-root "." \
  --map-image "c:/path/to/uploaded_map.png" \
  --project-id "client-project-id" \
  --client-name "Client Name" \
  --wait-kernel
```

## 1) Prepare project artifacts locally

Run project orchestrator:

```powershell
& "c:/Users/Aanshu Bhawsar/Desktop/DRL(Wildfire )/.venv/Scripts/python.exe" \
  "trainer/cv_pipeline/project_orchestrator.py" \
  --project-root "c:/Users/Aanshu Bhawsar/Desktop/DRL(Wildfire )" \
  --map-image "c:/path/to/uploaded_map.png" \
  --manifest "c:/path/to/project_manifest.json" \
  --seed-policy "c:/Users/Aanshu Bhawsar/Desktop/DRL(Wildfire )/trainer/models/dyna_q_policy.json" \
  --sim-experience "c:/Users/Aanshu Bhawsar/Desktop/DRL(Wildfire )/shared/sim_experience.json"
```

This creates:
- `shared/projects/<project_id>/...`
- `kaggle/dataset/projects/<project_id>/...`

## 2) Refresh Kaggle dataset payload

Keep top-level dataset files current:
- `kaggle/dataset/train_dyna_q.py`
- `kaggle/dataset/requirements-kaggle.txt`
- `kaggle/dataset/dataset-metadata.json`

Then version dataset:

```powershell
$env:KAGGLE_API_TOKEN="<TOKEN>"
& "c:/Users/Aanshu Bhawsar/Desktop/DRL(Wildfire )/.venv/Scripts/kaggle.exe" datasets version -p "kaggle/dataset" -m "project map update"
```

## 3) Push GPU kernel run

```powershell
$env:KAGGLE_API_TOKEN="<TOKEN>"
& "c:/Users/Aanshu Bhawsar/Desktop/DRL(Wildfire )/.venv/Scripts/kaggle.exe" kernels push -p "kaggle/kernel"
```

## 4) Monitor training and fetch outputs

```powershell
$env:KAGGLE_API_TOKEN="<TOKEN>"
& "c:/Users/Aanshu Bhawsar/Desktop/DRL(Wildfire )/.venv/Scripts/kaggle.exe" kernels status anshubhawsar/industrial-heat-drone-gpu-trainer
& "c:/Users/Aanshu Bhawsar/Desktop/DRL(Wildfire )/.venv/Scripts/kaggle.exe" kernels output anshubhawsar/industrial-heat-drone-gpu-trainer -p "kaggle/kernel/outputs"
```

## 5) Promote policy to simulator

Copy downloaded policy to:
- `trainer/models/dyna_q_policy.json`
- (optional) `kaggle/dataset/projects/<project_id>/dyna_q_policy.json` for warm-start next run

Then run frontend and validate drone behavior in the generated environment.
