# DRL Industrial Heat-Zone Drone

A realistic 3D web simulation and a Deep RL-style Dyna-Q training pipeline for a drone that finds active heat zones in an industrial environment.

## What is included

- 3D industrial simulation in browser (Three.js + Vite)
- Realistic lighting, fog, structures, smoke stacks, pipes, and dynamic heat plume effect
- Drone model with smooth movement and camera follow
- Active heat zones with temporal changes and noisy thermal sensing
- Large industrial environment and higher heat-zone density
- Interactive heat-zone controls: set zone count, add random zone, click-to-place zone
- Live DRL telemetry panel: state bins, action values, selected action, reward, and learning params
- Dyna-Q style training loop (Q-learning + planning updates)
- Shared JSON state format between simulator and trainer

## Project structure

- `web/`: Vite + TypeScript Three.js simulator
- `trainer/`: Python trainer using Dyna-Q approach
- `shared/`: shared config and optional exported trajectories

## Quick start

### 1) Web simulation

```bash
cd web
npm install
npm run dev
```

Open the local URL shown by Vite.

### 2) RL training

```bash
cd trainer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_dyna_q.py
```

If venv activation is not available in your shell, run directly with:

```bash
python trainer/train_dyna_q.py
```

A trained Q-table policy will be saved to `trainer/models/dyna_q_policy.json`.

## Deploying on Vercel

This repo is configured to deploy the web simulator from the `web/` folder.

1. Import the repository into Vercel.
2. Set the project root to the repository root.
3. Use the included `vercel.json` config at the root.
4. Vercel will run the web build with `cd web && npm run build` and publish `web/dist`.

The simulator is a single-page app, so the config also rewrites all routes to `index.html`.

## Exporting simulation experience for training

1. Start the web sim (`npm run dev` in `web/`)
2. Let the drone explore for 1-3 minutes
3. Press `E` in the simulator to export `sim_experience.json`
4. Move that file to `shared/sim_experience.json`

The trainer will automatically preload this experience before online Dyna-Q learning.

## Live controls in simulator

- Use the Heat Zones slider + Apply Zone Count to regenerate zones quickly
- Use Add Zone Mode and click on the ground to place custom hot spots
- Use Add 1 Random Zone for incremental difficulty
- Right panel shows DRL telemetry (state/action values and core parameters)

## Notes on "Q Dyno"

You asked for "q dyno" approaches, interpreted here as **Dyna-Q**:

- Real experience updates: standard Q-learning updates from environment transitions
- Planning updates: sampled updates from learned model `(s, a) -> (s', r)`
- Helps sample efficiency and faster convergence in sparse-reward search tasks

If you want, I can extend this to DQN + model-based imagination rollouts next.
