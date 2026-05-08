# Realism + Performance Spec (Mind-Blowing but Practical)

## Visual Realism Targets

1. PBR material families per semantic zone
- road: rough asphalt
- building shell: painted steel/concrete
- utility zones: metal + hazard decals
- rail: metal/oxide blend

2. Lighting strategy
- physically plausible daylight key + hemisphere fill
- subtle volumetric fog bands
- emissive accents around heat/utility zones

3. Environmental storytelling
- zone-specific props (pipes, tanks, stacks, loading bays)
- deterministic procedural placement with project seed
- decal overlays for wear, stains, directional markings

4. Camera and motion
- cinematic opening sweep + operational top-down mode
- dynamic exposure adaptation
- stabilized follow-camera with velocity-based lag

## Performance Optimization Targets

1. Geometry
- merged static geometry for buildings/floors
- instanced meshes for repeated assets
- LOD for distant props

2. Draw-call budget
- desktop target: < 500 draw calls
- laptop target: < 350 draw calls

3. Texture and shader policy
- atlas textures for repeating assets
- avoid expensive transparent materials in wide areas
- post FX toggles for low-end mode

4. Runtime checks
- maintain >= 55 FPS on baseline desktop
- avoid full scene rebuild on every tick
- event-driven updates for discovered zones and trails

## 2D -> 3D Conversion Quality Gates

1. Occupancy sanity
- blocked ratio between 5% and 70%
- at least one large connected free-space component

2. Scale sanity
- map scale conversion validated against known reference
- wall thickness clamped to realistic range

3. Navigation sanity
- spawn point in walkable cell
- at least one feasible route to target zones

## RL Coupling Requirements

1. Trainer and simulator use same occupancy grid semantics
2. Collision and boundary logic derived from occupancy map
3. Training report includes per-map metrics
4. Policy is promoted only if map-specific KPIs pass threshold
