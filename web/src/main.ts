import * as THREE from 'three';
import { jsPDF } from 'jspdf';
import './styles.css';
import { getTrainingApiBaseUrl } from './runtimeConfig';

/**
 * Autonomous Drone Fire Detection System
 * - Drone scans environment to DISCOVER hidden heat zones
 * - ML-based autonomous drone flight control
 * - Thermal anomaly detection as drone explores
 * - TSP shortest path solver for discovered zone visitation
 * - Real-time thermal imaging and pathfinding
 */

// ============================================
// APP CHROME
// ============================================
function setupAppUI(): void {
  const docBtn = document.getElementById('doc-btn') as HTMLButtonElement;
  
  if (docBtn) {
    docBtn.addEventListener('click', () => {
      window.open('/new.html', '_blank', 'noopener,noreferrer');
    });
  }
}

type HeatZone = {
  id: number;
  x: number;
  z: number;
  intensity: number;
  smell: number;
  originalIntensity: number;
  originalSmell: number;
  found: boolean;
};

type WayPoint = {
  x: number;
  y: number;
  z: number;
};

type ThermalReading = {
  x: number;
  z: number;
  intensity: number;
  timestamp: number;
};

type DroneObservation = {
  relativePosition: { x: number; z: number };
  thermalGradient: number;
};

type RLAction = {
  dvx: number;
  dvz: number;
};

type CustomEnvironmentObject = {
  type: string;
  height: number;
  x: number;
  z: number;
  width?: number;
  depth?: number;
  mesh?: THREE.Object3D;
};

type HybridPolicyConfig = {
  grid_x?: number;
  grid_z?: number;
  t_bins?: number;
  coverage_bins?: number;
};

type HybridPolicyPayload = {
  algorithm?: string;
  best_actions?: number[][][][];
  config?: HybridPolicyConfig;
};

const canvas = document.getElementById('scene') as HTMLCanvasElement;
const metrics = document.getElementById('metrics') as HTMLDivElement;
const controls = document.getElementById('controls') as HTMLDivElement;
const drlMetrics = document.getElementById('drl-metrics') as HTMLDivElement;
let droneStatusEl: HTMLElement | null = null;
let missionCountdownEl: HTMLElement | null = null;
let zonesDiscoveredEl: HTMLElement | null = null;
let zonesVisitedEl: HTMLElement | null = null;
let distanceValEl: HTMLElement | null = null;
let speedValEl: HTMLElement | null = null;

const WORLD_SIZE = 120;
const SCAN_RADIUS = 30;
const SCAN_INTERVAL = 0.12;
const MAX_DRONE_SPEED = 40;
let currentCameraView: 'topdown' | 'side' | 'follow' | 'isometric' | 'freelook' = 'freelook';
let heatZoneCount = 3;
let buildingFields: { x: number; z: number; radius: number }[] = [];
let customBuildingFields: { x: number; z: number; radius: number }[] = [];

// Hidden heat zones (not known to drone - must be discovered)
let HIDDEN_HEAT_ZONES: HeatZone[] = [];

let droneState = {
  position: new THREE.Vector3(0, 8, 0),
  velocity: new THREE.Vector3(0, 0, 0),
  distanceTraveled: 0,
  maxSpeed: 15.0,
  acceleration: 3.5,
  isFlying: false,
  missionComplete: false,
  lastScanTime: 0,
};

// Discovered zones (drone finds these)
let discoveredZones: HeatZone[] = [];
let thermalReadings: ThermalReading[] = [];
let droneTrail: THREE.Vector3[] = [];
let isLaunched = false;
let explorationMode = true;
let lastTelemetryUpdate = 0;
let currentThermalSignal = 0;
let lastThermalSignal = 0;
let cumulativeReward = 0;
let episodeSteps = 0;
let lastCollision = false;
let discoveredZonesVisualDirty = true;
let epsilon = 0.28;
let policyStatus: 'loading' | 'ready' | 'unavailable' = 'loading';
let hybridPolicy: HybridPolicyPayload | null = null;
let selectedPolicyPath = '/models/hybrid_drl_explorer_policy.json';
let policyGridX = 36;
let policyGridZ = 36;
let policyTBins = 10;
let policyCoverageBins = 8;
let thermalBinSize = 12; // degrees per thermal bin (configurable)
let thermalOffset = 0; // degrees shift applied before binning
let coverageCells = new Set<string>();
let recentGridCells: string[] = [];
let cellVisitCounts = new Map<string, number>();
let sameActionStreak = 0;
let lastPolicyActionIdx = -1;
let lowMotionStreak = 0;
let loopEscapeTicks = 0;
let collisionRecoveryTimer = 0;
let missionTimeRemaining = 300;
let missionActive = false;
let missionFinalized = false;
let tspRoute: WayPoint[] = [];
let tspGraphZoom = 1;
let activeTspPoints: HeatZone[] = [];
let heldAction: RLAction = { dvx: 0, dvz: 0 };
let actionDecisionTimer = 0;
const smoothedAction = new THREE.Vector2(0, 0);

const MISSION_DURATION_SECONDS = 300;

const EPSILON_MIN = 0.05;
const EPSILON_DECAY = 0.999;
const REWARD_RESET_THRESHOLD = -120;
const POLICY_WORLD_LIMIT = WORLD_SIZE * 1.1;
const LOOP_WINDOW = 30;
const LOOP_CELL_REPEAT_THRESHOLD = 10;
const SAME_ACTION_THRESHOLD = 14;
const LOW_MOTION_THRESHOLD = 14;
const LOOP_ESCAPE_STEPS = 36;
const DRONE_BOUNDARY_LIMIT = WORLD_SIZE * 1.08;
const VISUAL_REFRESH_INTERVAL = 0.09;
const SPAWN_SAFE_RADIUS = 28;
const COLLISION_RECOVERY_COOLDOWN = 0.6;
const ACTION_DECISION_INTERVAL = 0.14;
const ACTION_SMOOTHING = 0.2;
const VELOCITY_DEADZONE = 0.28;
const HYBRID_ACTION_DELTAS: { x: number; z: number }[] = [
  { x: 1, z: 0 },
  { x: -1, z: 0 },
  { x: 0, z: 1 },
  { x: 0, z: -1 },
  { x: 1, z: 1 },
  { x: 1, z: -1 },
  { x: -1, z: 1 },
  { x: -1, z: -1 },
  { x: 0, z: 0 },
];

const API_BASE_URL = getTrainingApiBaseUrl();

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x081018);
scene.fog = new THREE.FogExp2(0x0a1a24, 0.0055);

// Create custom environment objects group
const customObjectsGroup = new THREE.Group();
scene.add(customObjectsGroup);

// Raycaster for click-to-place
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.15));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = false;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.6;
renderer.sortObjects = false;

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 500);
camera.position.set(50, 40, 60);
camera.lookAt(0, 0, 0);

function updateCameraView(): void {
  if (currentCameraView === 'topdown') {
    camera.position.set(0, 100, 0);
    camera.lookAt(0, 0, 0);
  } else if (currentCameraView === 'side') {
    camera.position.set(150, 30, 0);
    camera.lookAt(0, 10, 0);
  } else if (currentCameraView === 'follow') {
    const speedFactor = Math.max(0.6, Math.min(1.8, droneState.velocity.length() / 5));
    const velocityDir = droneState.velocity.lengthSq() > 0.2 ? droneState.velocity.clone().normalize() : new THREE.Vector3(0, 0, 1);
    const offset = velocityDir.multiplyScalar(-24 * speedFactor).add(new THREE.Vector3(0, 13, 0));
    const targetPos = droneState.position.clone().add(offset);
    camera.position.lerp(targetPos, 0.1);
    camera.lookAt(droneState.position.x, droneState.position.y, droneState.position.z);
  } else if (currentCameraView === 'isometric') {
    camera.position.set(70, 60, 70);
    camera.lookAt(0, 15, 0);
  } else if (currentCameraView === 'freelook') {
    const angle = clock.getElapsedTime() * 0.06;
    camera.position.set(60 * Math.cos(angle), 40, 80 * Math.sin(angle));
    camera.lookAt(0, 10, 0);
  }
}

// Lighting
const ambientLight = new THREE.AmbientLight(0x37536a, 0.92);
scene.add(ambientLight);

const hemiLight = new THREE.HemisphereLight(0x9be6ff, 0x182230, 0.66);
scene.add(hemiLight);

const dirLight = new THREE.DirectionalLight(0xffe7c4, 1.95);
dirLight.position.set(72, 62, 28);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
dirLight.shadow.camera.far = 250;
scene.add(dirLight);

const rimLightA = new THREE.PointLight(0x00e1ff, 1.35, 260);
rimLightA.position.set(-70, 26, -55);
scene.add(rimLightA);

const rimLightB = new THREE.PointLight(0xff6a36, 1.2, 220);
rimLightB.position.set(55, 20, 68);
scene.add(rimLightB);

function createHeatZones(count: number): HeatZone[] {
  const zones: HeatZone[] = [];
  const minSpacing = 18;

  for (let i = 0; i < count; i++) {
    let attempts = 0;
    let x = 0;
    let z = 0;
    let valid = false;

    while (!valid && attempts < 50) {
      x = (Math.random() - 0.5) * WORLD_SIZE * 1.5;
      z = (Math.random() - 0.5) * WORLD_SIZE * 1.5;
      valid = zones.every((zone) => Math.hypot(zone.x - x, zone.z - z) >= minSpacing);
      attempts++;
    }

    zones.push({
      id: i + 1,
      x,
      z,
      intensity: 0.7 + Math.random() * 0.28,
      smell: 0.7 + Math.random() * 0.28,
      originalIntensity: 0,
      originalSmell: 0,
      found: false,
    });
    zones[zones.length - 1].originalIntensity = zones[zones.length - 1].intensity;
    zones[zones.length - 1].originalSmell = zones[zones.length - 1].smell;
  }

  return zones;
}

function resetHeatZones(count: number): void {
  heatZoneCount = count;
  HIDDEN_HEAT_ZONES = createHeatZones(count);
  discoveredZones = [];
  thermalReadings = [];
}

resetHeatZones(heatZoneCount);

// Ground
const groundGeo = new THREE.PlaneGeometry(WORLD_SIZE * 2.5, WORLD_SIZE * 2.5, 48, 48);
const groundMat = new THREE.MeshStandardMaterial({
  color: 0x17242c,
  roughness: 0.88,
  metalness: 0.16,
  emissive: 0x081822,
  emissiveIntensity: 0.38,
});
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// Grid
const grid = new THREE.GridHelper(WORLD_SIZE * 2.5, 60, 0x2b6f88, 0x102839);
(grid.material as THREE.Material).opacity = 0.2;
(grid.material as THREE.Material).transparent = true;
scene.add(grid);

const runwayGroup = new THREE.Group();
scene.add(runwayGroup);

for (let i = -3; i <= 3; i++) {
  const stripe = new THREE.Mesh(
    new THREE.PlaneGeometry(WORLD_SIZE * 2.1, 1.2),
    new THREE.MeshBasicMaterial({
      color: i === 0 ? 0xff8a3d : 0x2a9fd6,
      transparent: true,
      opacity: i === 0 ? 0.36 : 0.16,
    }),
  );
  stripe.rotation.x = -Math.PI / 2;
  stripe.position.set(0, 0.08, i * 18);
  runwayGroup.add(stripe);
}

const enclosureGroup = new THREE.Group();
scene.add(enclosureGroup);

function addEnclosureWall(width: number, height: number, depth: number, x: number, z: number, rotationY: number): void {
  const wall = new THREE.Mesh(
    new THREE.BoxGeometry(width, height, depth),
    new THREE.MeshStandardMaterial({
      color: 0x1f3340,
      roughness: 0.78,
      metalness: 0.34,
      emissive: 0x06111b,
      emissiveIntensity: 0.16,
    }),
  );
  wall.position.set(x, height / 2, z);
  wall.rotation.y = rotationY;
  wall.castShadow = true;
  wall.receiveShadow = true;
  enclosureGroup.add(wall);
}

addEnclosureWall(WORLD_SIZE * 2.1, 18, 3.5, 0, -WORLD_SIZE * 1.18, 0);
addEnclosureWall(WORLD_SIZE * 2.1, 18, 3.5, 0, WORLD_SIZE * 1.18, 0);
addEnclosureWall(3.5, 18, WORLD_SIZE * 2.1, -WORLD_SIZE * 1.18, 0, 0);
addEnclosureWall(3.5, 18, WORLD_SIZE * 2.1, WORLD_SIZE * 1.18, 0, 0);

const pipeGroup = new THREE.Group();
scene.add(pipeGroup);

for (let i = 0; i < 6; i++) {
  const pipe = new THREE.Mesh(
    new THREE.CylinderGeometry(0.6, 0.6, WORLD_SIZE * 1.1, 10),
    new THREE.MeshStandardMaterial({
      color: 0x4f6b7d,
      roughness: 0.46,
      metalness: 0.52,
      emissive: 0x0a1b28,
      emissiveIntensity: 0.16,
    }),
  );
  pipe.rotation.z = Math.PI / 2;
  pipe.position.set((Math.random() - 0.5) * WORLD_SIZE * 1.2, 22 + Math.random() * 5, (Math.random() - 0.5) * WORLD_SIZE * 1.2);
  pipe.castShadow = true;
  pipeGroup.add(pipe);

  const pipeGlow = new THREE.PointLight(i % 2 === 0 ? 0x2dd5ff : 0xff8a3d, 0.45, 45);
  pipeGlow.position.copy(pipe.position);
  pipeGlow.position.y += 2;
  pipeGroup.add(pipeGlow);
}

const haze = new THREE.FogExp2(0x09131a, 0.0048);
scene.fog = haze;

// Industrial structures
const industryGroup = new THREE.Group();
scene.add(industryGroup);

const industryPalette = [0x3a5f78, 0x436f8b, 0x447a6e, 0x746b4d, 0x5e4f74, 0x7e4b56];
const INDUSTRY_SPREAD = WORLD_SIZE * 2.35;

// Main complex blocks
for (let i = 0; i < 24; i++) {
  let px = 0;
  let pz = 0;
  let placed = false;
  for (let attempt = 0; attempt < 40; attempt++) {
    px = (Math.random() - 0.5) * INDUSTRY_SPREAD;
    pz = (Math.random() - 0.5) * INDUSTRY_SPREAD;
    const tooCloseToSpawn = Math.hypot(px, pz) < SPAWN_SAFE_RADIUS;
    if (!tooCloseToSpawn) {
      placed = true;
      break;
    }
  }
  if (!placed) {
    continue;
  }

  const width = 12 + Math.random() * 14;
  const height = 10 + Math.random() * 26;
  const depth = 10 + Math.random() * 14;
  const baseColor = industryPalette[i % industryPalette.length];
  const struct = new THREE.Mesh(
    new THREE.BoxGeometry(width, height, depth),
    new THREE.MeshStandardMaterial({
      color: baseColor,
      roughness: 0.52,
      metalness: 0.48,
      emissive: 0x0b2032,
      emissiveIntensity: 0.12,
    }),
  );
  struct.position.set(px, height / 2 + 0.4, pz);
  struct.castShadow = true;
  struct.receiveShadow = true;
  industryGroup.add(struct);

  buildingFields.push({
    x: px,
    z: pz,
    radius: Math.max(width, depth) * 0.72 + 6,
  });

  const roofRing = new THREE.Mesh(
    new THREE.RingGeometry(Math.min(width, depth) * 0.16, Math.min(width, depth) * 0.35, 20),
    new THREE.MeshBasicMaterial({ color: 0x19d4ff, transparent: true, opacity: 0.18 }),
  );
  roofRing.rotation.x = -Math.PI / 2;
  roofRing.position.set(px, height + 0.65, pz);
  industryGroup.add(roofRing);

  const sideTank = new THREE.Mesh(
    new THREE.CylinderGeometry(1.8, 1.8, Math.max(6, height * 0.45), 14),
    new THREE.MeshStandardMaterial({ color: 0x8ea3b4, roughness: 0.38, metalness: 0.58, emissive: 0x1e2328, emissiveIntensity: 0.1 }),
  );
  sideTank.position.set(px + width * 0.34, Math.max(6, height * 0.45) / 2 + 0.4, pz - depth * 0.34);
  sideTank.castShadow = true;
  industryGroup.add(sideTank);

  const catwalk = new THREE.Mesh(
    new THREE.BoxGeometry(width * 0.78, 0.35, 1.4),
    new THREE.MeshStandardMaterial({ color: 0x94b7c9, roughness: 0.42, metalness: 0.55 }),
  );
  catwalk.position.set(px, height * 0.62, pz + depth * 0.52);
  industryGroup.add(catwalk);

  const warningLamp = new THREE.PointLight(i % 2 === 0 ? 0xff8a3d : 0x24d2ff, 0.7, 60);
  warningLamp.position.set(px, height + 5, pz);
  industryGroup.add(warningLamp);
}

// Colorful container yards for visual richness
const containerPalette = [0xf05a50, 0x22b8cf, 0x8bc34a, 0xffc857, 0xa56eff, 0xff7f50];
for (let i = 0; i < 40; i++) {
  let px = 0;
  let pz = 0;
  let placed = false;
  for (let attempt = 0; attempt < 30; attempt++) {
    px = (Math.random() - 0.5) * INDUSTRY_SPREAD;
    pz = (Math.random() - 0.5) * INDUSTRY_SPREAD;
    if (Math.hypot(px, pz) >= SPAWN_SAFE_RADIUS * 0.9) {
      placed = true;
      break;
    }
  }
  if (!placed) {
    continue;
  }

  const c = new THREE.Mesh(
    new THREE.BoxGeometry(4.4, 2.4, 2.2),
    new THREE.MeshStandardMaterial({
      color: containerPalette[i % containerPalette.length],
      roughness: 0.5,
      metalness: 0.38,
      emissive: 0x111111,
      emissiveIntensity: 0.08,
    }),
  );
  c.position.set(px, 1.2 + (i % 3) * 2.5, pz);
  c.castShadow = true;
  c.receiveShadow = true;
  industryGroup.add(c);
}

// Discovered heat zones visualization
const discoveredZonesGroup = new THREE.Group();
scene.add(discoveredZonesGroup);
const discoveredZoneVisualCache = new Map<number, { circle: THREE.Mesh; light: THREE.PointLight; sprite: THREE.Sprite }>();

function updateDiscoveredZonesVisuals(): void {
  if (!discoveredZonesVisualDirty) {
    return;
  }

  discoveredZonesGroup.clear();

  for (const zone of discoveredZones) {
    let cached = discoveredZoneVisualCache.get(zone.id);
    if (!cached) {
      const circle = new THREE.Mesh(
        new THREE.CircleGeometry(10, 16),
        new THREE.MeshBasicMaterial({
          color: zone.found ? 0x00ff99 : 0xffaa00,
          transparent: true,
          opacity: 0.3,
        }),
      );
      circle.rotation.x = -Math.PI / 2;

      const light = new THREE.PointLight(zone.found ? 0x00ff99 : 0xff6600, 1.6, 42);

      const spriteCanvas = document.createElement('canvas');
      spriteCanvas.width = 64;
      spriteCanvas.height = 64;
      const spriteCtx = spriteCanvas.getContext('2d')!;
      spriteCtx.fillStyle = zone.found ? '#00ff99' : '#ffaa00';
      spriteCtx.font = 'Bold 40px Arial';
      spriteCtx.textAlign = 'center';
      spriteCtx.fillText(`Z${zone.id}`, 32, 45);
      const spriteTex = new THREE.CanvasTexture(spriteCanvas);
      const spriteMat = new THREE.SpriteMaterial({ map: spriteTex });
      const sprite = new THREE.Sprite(spriteMat);
      sprite.scale.set(12, 12, 1);

      cached = { circle, light, sprite };
      discoveredZoneVisualCache.set(zone.id, cached);
    }

    (cached.circle.material as THREE.MeshBasicMaterial).color.set(zone.found ? 0x00ff99 : 0xffaa00);
    cached.circle.position.set(zone.x, 0.05, zone.z);
    cached.light.color.set(zone.found ? 0x00ff99 : 0xff6600);
    cached.light.position.set(zone.x, 18, zone.z);
    cached.sprite.position.set(zone.x, 28, zone.z);

    discoveredZonesGroup.add(cached.circle);
    discoveredZonesGroup.add(cached.light);
    discoveredZonesGroup.add(cached.sprite);
  }

  discoveredZonesVisualDirty = false;
}

// Thermal scan visualization (cone showing scan area)
const scanVisualsGroup = new THREE.Group();
scene.add(scanVisualsGroup);
const scanGeo = new THREE.SphereGeometry(SCAN_RADIUS, 8, 8);
const scanMat = new THREE.MeshBasicMaterial({
  color: 0x00ffff,
  transparent: true,
  opacity: 0.06,
  wireframe: true,
});
const scanMesh = new THREE.Mesh(scanGeo, scanMat);
scanVisualsGroup.add(scanMesh);

function visualizeThermalScan(): void {
  scanMesh.visible = droneState.isFlying;
  scanMesh.position.copy(droneState.position);
  scanMesh.scale.setScalar(1 + Math.sin(clock.getElapsedTime() * 4) * 0.01);
}

// Drone model
const droneGroup = new THREE.Group();
scene.add(droneGroup);

const droneCore = new THREE.Mesh(
  new THREE.CylinderGeometry(1.8, 2.3, 1.6, 24),
  new THREE.MeshStandardMaterial({
    color: 0x93e4ff,
    emissive: 0x18d3ff,
    emissiveIntensity: 0.32,
    roughness: 0.28,
    metalness: 0.8,
  }),
);
droneCore.castShadow = true;
droneGroup.add(droneCore);

const droneCanopy = new THREE.Mesh(
  new THREE.SphereGeometry(1.1, 20, 20),
  new THREE.MeshStandardMaterial({
    color: 0x1e3048,
    emissive: 0x2ca7ff,
    emissiveIntensity: 0.26,
    roughness: 0.08,
    metalness: 0.88,
    transparent: true,
    opacity: 0.82,
  }),
);
droneCanopy.position.y = 0.78;
droneGroup.add(droneCanopy);

const droneRing = new THREE.Mesh(
  new THREE.TorusGeometry(2.55, 0.18, 14, 48),
  new THREE.MeshStandardMaterial({
    color: 0x00ffa8,
    emissive: 0x00ffa8,
    emissiveIntensity: 0.65,
    roughness: 0.22,
    metalness: 0.58,
  }),
);
droneRing.rotation.x = Math.PI / 2;
droneRing.position.y = 0.12;
droneGroup.add(droneRing);

const droneNose = new THREE.Mesh(
  new THREE.ConeGeometry(0.34, 1.3, 16),
  new THREE.MeshStandardMaterial({ color: 0xffb55a, emissive: 0xff6a3d, emissiveIntensity: 0.45 }),
);
droneNose.rotation.x = Math.PI / 2;
droneNose.position.set(0, 0.14, 2.5);
droneGroup.add(droneNose);

const propellers: THREE.Mesh[] = [];
const armPositions = [
  new THREE.Vector3(2.8, 0.45, 2.8),
  new THREE.Vector3(-2.8, 0.45, 2.8),
  new THREE.Vector3(2.8, 0.45, -2.8),
  new THREE.Vector3(-2.8, 0.45, -2.8),
];

for (const armPos of armPositions) {
  const arm = new THREE.Mesh(
    new THREE.CylinderGeometry(0.14, 0.14, 4.4, 12),
    new THREE.MeshStandardMaterial({ color: 0x6cb9d8, emissive: 0x1f4d66, emissiveIntensity: 0.28 }),
  );
  arm.position.copy(armPos.clone().multiplyScalar(0.5));
  arm.lookAt(armPos.x, armPos.y, armPos.z);
  arm.rotateX(Math.PI / 2);
  arm.castShadow = true;
  droneGroup.add(arm);

  const motor = new THREE.Mesh(
    new THREE.CylinderGeometry(0.44, 0.52, 0.38, 16),
    new THREE.MeshStandardMaterial({ color: 0x8ecff5, emissive: 0x1f8ac4, emissiveIntensity: 0.4, metalness: 0.75 }),
  );
  motor.position.copy(armPos);
  motor.castShadow = true;
  droneGroup.add(motor);

  const prop = new THREE.Mesh(
    new THREE.TorusGeometry(0.95, 0.06, 10, 32),
    new THREE.MeshStandardMaterial({ color: 0xff5fa2, emissive: 0xff3a8d, emissiveIntensity: 0.72 }),
  );
  prop.position.copy(armPos.clone().add(new THREE.Vector3(0, 0.26, 0)));
  prop.rotation.x = Math.PI / 2;
  propellers.push(prop);
  droneGroup.add(prop);
}

const droneBeacon = new THREE.PointLight(0x00ffd9, 1.8, 26);
droneBeacon.position.set(0, 1.8, 0);
droneGroup.add(droneBeacon);

// Drone trail
const trailLines: THREE.Line[] = [];
let trailTimer = 0;

function updateDroneTrail(dt: number): void {
  trailTimer += dt;
  if (trailTimer < 0.14) {
    return;
  }
  trailTimer = 0;

  if (droneState.velocity.length() < 0.3) {
    return;
  }

  droneTrail.push(droneState.position.clone());
  if (droneTrail.length > 50) droneTrail.shift();

  // Update trail visualization
  if (trailLines.length > 0) {
    scene.remove(trailLines[0]);
    trailLines.shift();
  }

  if (droneTrail.length > 1) {
    const geo = new THREE.BufferGeometry().setFromPoints(droneTrail);
    const mat = new THREE.LineBasicMaterial({ color: 0x00ffdd, transparent: true, opacity: 0.45 });
    const line = new THREE.Line(geo, mat);
    scene.add(line);
    trailLines.push(line);
  }
}

// Thermal scanning: detect nearby heat zones
function performThermalScan(): number {
  const now = Date.now() / 1000;

  if (now - droneState.lastScanTime < SCAN_INTERVAL) {
    return currentThermalSignal;
  }
  droneState.lastScanTime = now;
  let signal = 0;

  // Scan for hidden heat zones
  for (const hiddenZone of HIDDEN_HEAT_ZONES) {
    if (hiddenZone.found) {
      continue;
    }

    const distance = droneState.position.distanceTo(new THREE.Vector3(hiddenZone.x, 8, hiddenZone.z));
    const influence = hiddenZone.intensity * Math.exp(-(distance * distance) / (2 * (SCAN_RADIUS * 0.8) * (SCAN_RADIUS * 0.8)));
    signal += influence;

    if (distance < SCAN_RADIUS) {
      // Store thermal reading (limit history for performance)
      thermalReadings.push({
        x: hiddenZone.x,
        z: hiddenZone.z,
        intensity: hiddenZone.intensity,
        timestamp: now,
      });
      if (thermalReadings.length > 100) thermalReadings.shift();

      // Check if zone already discovered
      const existing = discoveredZones.find((z) => Math.hypot(z.x - hiddenZone.x, z.z - hiddenZone.z) < 5);
      if (!existing) {
        hiddenZone.found = true;
        hiddenZone.intensity = 0;
        hiddenZone.smell = 0;
        discoveredZones.push({ ...hiddenZone });
        discoveredZonesVisualDirty = true;
        console.log(`Zone ${hiddenZone.id} DISCOVERED at (${hiddenZone.x}, ${hiddenZone.z})`);
      } else {
        existing.found = true;
        existing.intensity = 0;
        existing.smell = 0;
        existing.originalIntensity = existing.originalIntensity ?? hiddenZone.originalIntensity;
        existing.originalSmell = existing.originalSmell ?? hiddenZone.originalSmell;
        discoveredZonesVisualDirty = true;
      }
    }
  }

  currentThermalSignal = signal;
  return signal;
}

function getDroneObservation(): DroneObservation {
  const relativeX = droneState.position.x / WORLD_SIZE;
  const relativeZ = droneState.position.z / WORLD_SIZE;
  const gradientRaw = currentThermalSignal - lastThermalSignal;
  const thermalGradient = Math.max(-1, Math.min(1, gradientRaw * 5));

  return {
    relativePosition: { x: relativeX, z: relativeZ },
    thermalGradient,
  };
}

function computeBuildingAvoidance(position: THREE.Vector3): THREE.Vector3 {
  const avoidance = new THREE.Vector3(0, 0, 0);

  // Combine both procedural and custom buildings
  const allBuildings = [...buildingFields, ...customBuildingFields];

  for (const building of allBuildings) {
    const dx = position.x - building.x;
    const dz = position.z - building.z;
    const distance = Math.hypot(dx, dz);
    const influence = building.radius + 10;

    if (distance > 0.001 && distance < influence) {
      const strength = ((influence - distance) / influence) ** 2 * 22;
      avoidance.x += (dx / distance) * strength;
      avoidance.z += (dz / distance) * strength;
    }
  }

  return avoidance;
}
function isPointBlockedByBuilding(x: number, z: number, buffer = 8): boolean {
  const allBuildings = [...buildingFields, ...customBuildingFields];
  return allBuildings.some((building) => Math.hypot(x - building.x, z - building.z) < building.radius + buffer);
}

function clampIndex(v: number, maxExclusive: number): number {
  return Math.max(0, Math.min(maxExclusive - 1, v));
}

function worldToPolicyGrid(x: number, z: number): { gx: number; gz: number } {
  const nx = (x + POLICY_WORLD_LIMIT) / (POLICY_WORLD_LIMIT * 2);
  const nz = (z + POLICY_WORLD_LIMIT) / (POLICY_WORLD_LIMIT * 2);
  return {
    gx: clampIndex(Math.floor(nx * policyGridX), policyGridX),
    gz: clampIndex(Math.floor(nz * policyGridZ), policyGridZ),
  };
}

function markCoverageCell(): void {
  const grid = worldToPolicyGrid(droneState.position.x, droneState.position.z);
  const key = `${grid.gx},${grid.gz}`;
  coverageCells.add(key);
  recentGridCells.push(key);
  if (recentGridCells.length > LOOP_WINDOW) {
    recentGridCells.shift();
  }
  cellVisitCounts.set(key, (cellVisitCounts.get(key) ?? 0) + 1);
}

function isLoopingPattern(actionIdx: number): boolean {
  const currentGrid = worldToPolicyGrid(droneState.position.x, droneState.position.z);
  const key = `${currentGrid.gx},${currentGrid.gz}`;
  const recentRepeats = recentGridCells.reduce((acc, cell) => acc + (cell === key ? 1 : 0), 0);
  const totalVisits = cellVisitCounts.get(key) ?? 0;

  if (actionIdx === lastPolicyActionIdx) {
    sameActionStreak++;
  } else {
    sameActionStreak = 0;
    lastPolicyActionIdx = actionIdx;
  }

  return (
    recentRepeats >= LOOP_CELL_REPEAT_THRESHOLD
    || totalVisits >= LOOP_CELL_REPEAT_THRESHOLD + 2
    || sameActionStreak >= SAME_ACTION_THRESHOLD
    || lowMotionStreak >= LOW_MOTION_THRESHOLD
  );
}

function pickEscapeActionIndex(preferredActionIdx: number): number {
  const shuffled = [0, 1, 2, 3, 4, 5, 6, 7].sort(() => Math.random() - 0.5);

  for (const idx of shuffled) {
    if (idx === preferredActionIdx) continue;
    const delta = HYBRID_ACTION_DELTAS[idx];
    const trialX = droneState.position.x + delta.x * 7;
    const trialZ = droneState.position.z + delta.z * 7;
    if (Math.abs(trialX) > POLICY_WORLD_LIMIT || Math.abs(trialZ) > POLICY_WORLD_LIMIT) continue;
    if (!isPointBlockedByBuilding(trialX, trialZ, 6.5)) {
      return idx;
    }
  }

  return (preferredActionIdx + 3) % 8;
}

function getPolicyActionIndex(): number | null {
  if (!hybridPolicy?.best_actions) {
    return null;
  }

  const { gx, gz } = worldToPolicyGrid(droneState.position.x, droneState.position.z);
  const thermalBin = clampIndex(Math.floor((currentThermalSignal + thermalOffset) / Math.max(1, thermalBinSize)), policyTBins);
  const coverageRatio = coverageCells.size / Math.max(1, policyGridX * policyGridZ);
  const coverageBin = clampIndex(Math.floor(coverageRatio * policyCoverageBins), policyCoverageBins);
  const action = hybridPolicy.best_actions[gx]?.[gz]?.[thermalBin]?.[coverageBin];

  return typeof action === 'number' ? clampIndex(action, HYBRID_ACTION_DELTAS.length) : null;
}

function actionIndexToVelocityDelta(actionIdx: number, observation: DroneObservation): RLAction {
  const delta = HYBRID_ACTION_DELTAS[clampIndex(actionIdx, HYBRID_ACTION_DELTAS.length)];
  if (delta.x === 0 && delta.z === 0) {
    return {
      dvx: -observation.relativePosition.x * 0.2,
      dvz: -observation.relativePosition.z * 0.2,
    };
  }

  const norm = Math.max(1, Math.hypot(delta.x, delta.z));
  return {
    dvx: (delta.x / norm) * 0.92,
    dvz: (delta.z / norm) * 0.92,
  };
}

function computeTspRoute(points: WayPoint[]): WayPoint[] {
  if (points.length === 0) {
    return [];
  }

  const remaining = points.map((point) => ({ ...point }));
  const route: WayPoint[] = [{ x: 0, y: 8, z: 0 }];
  let current = route[0];

  while (remaining.length > 0) {
    let bestIndex = 0;
    let bestDistance = Number.POSITIVE_INFINITY;

    for (let i = 0; i < remaining.length; i++) {
      const candidate = remaining[i];
      const distance = Math.hypot(candidate.x - current.x, candidate.z - current.z);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = i;
      }
    }

    current = remaining.splice(bestIndex, 1)[0];
    route.push(current);
  }

  route.push({ x: 0, y: 8, z: 0 });
  return route;
}

function getTspPopup(): HTMLDivElement {
  let popup = document.getElementById('tsp-popup') as HTMLDivElement | null;
  if (popup) {
    return popup;
  }

  popup = document.createElement('div');
  popup.id = 'tsp-popup';
  popup.className = 'tsp-popup hidden';
  popup.innerHTML = `
    <div class="tsp-popup-header">
      <div>
        <div class="tsp-popup-title">TSP Route Report</div>
        <div class="tsp-popup-subtitle">Heat zones discovered and route solution</div>
      </div>
      <button id="tsp-close-btn">Close</button>
    </div>
    <div style="display: flex; align-items: center; gap: 10px; margin: 10px 0 8px; color: #bcefff;">
      <label for="tsp-zoom-slider" style="white-space: nowrap;">Graph Fit</label>
      <input id="tsp-zoom-slider" type="range" min="0.6" max="1.4" step="0.05" value="1" style="width: 100%;">
      <span id="tsp-zoom-value" style="min-width: 48px; text-align: right;">1.00x</span>
    </div>
    <canvas id="tsp-graph" width="760" height="520"></canvas>
    <div id="tsp-summary" class="tsp-summary"></div>
  `;
  document.body.appendChild(popup);

  const closeBtn = popup.querySelector('#tsp-close-btn') as HTMLButtonElement | null;
  closeBtn?.addEventListener('click', () => {
    popup?.classList.add('hidden');
  });

  const zoomSlider = popup.querySelector('#tsp-zoom-slider') as HTMLInputElement | null;
  const zoomValue = popup.querySelector('#tsp-zoom-value') as HTMLSpanElement | null;
  zoomSlider?.addEventListener('input', () => {
    tspGraphZoom = Math.max(0.6, Math.min(1.4, parseFloat(zoomSlider.value) || 1));
    if (zoomValue) {
      zoomValue.textContent = `${tspGraphZoom.toFixed(2)}x`;
    }
    if (tspRoute.length > 0 && activeTspPoints.length > 0) {
      drawTspGraph(tspRoute, activeTspPoints);
    }
  });

  return popup;
}

function drawTspGraph(route: WayPoint[], points: HeatZone[]): void {
  const popup = getTspPopup();
  const canvasEl = popup.querySelector('#tsp-graph') as HTMLCanvasElement | null;
  const summaryEl = popup.querySelector('#tsp-summary') as HTMLDivElement | null;
  if (!canvasEl || !summaryEl) {
    return;
  }

  const ctx = canvasEl.getContext('2d');
  if (!ctx) {
    return;
  }

  renderTspCanvas(canvasEl, route, points);

  summaryEl.innerHTML = [
    `Discovered Heat Zones: ${points.length}`,
    `TSP Route Nodes: ${Math.max(0, route.length - 2)}`,
    `Mission Time Left: ${Math.max(0, Math.ceil(missionTimeRemaining))} sec`,
  ].join('<br/>');
}

function renderTspCanvas(canvasEl: HTMLCanvasElement, route: WayPoint[], points: HeatZone[]): void {
  const ctx = canvasEl.getContext('2d');
  if (!ctx) {
    return;
  }

  const allPoints = [{ x: 0, z: 0 }, ...points.map((zone) => ({ x: zone.x, z: zone.z }))];
  const xs = allPoints.map((point) => point.x);
  const zs = allPoints.map((point) => point.z);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minZ = Math.min(...zs);
  const maxZ = Math.max(...zs);
  const width = canvasEl.width;
  const height = canvasEl.height;
  const baseScaleX = (width - 84) / Math.max(1, maxX - minX);
  const baseScaleZ = (height - 84) / Math.max(1, maxZ - minZ);
  const baseScale = Math.min(baseScaleX, baseScaleZ);
  const scale = baseScale * tspGraphZoom;

  const contentWidth = Math.max(1, (maxX - minX) * scale);
  const contentHeight = Math.max(1, (maxZ - minZ) * scale);
  const offsetX = (width - contentWidth) / 2 - minX * scale;
  const offsetZ = (height - contentHeight) / 2 - minZ * scale;

  const mapPoint = (pt: { x: number; z: number }) => ({
    x: offsetX + pt.x * scale,
    y: height - (offsetZ + pt.z * scale),
  });

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#07131e';
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = 'rgba(71, 225, 255, 0.12)';
  ctx.lineWidth = 1;
  for (let i = 0; i < 12; i++) {
    const x = 32 + (i / 11) * (width - 64);
    const y = 32 + (i / 11) * (height - 64);
    ctx.beginPath();
    ctx.moveTo(x, 32);
    ctx.lineTo(x, height - 32);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(32, y);
    ctx.lineTo(width - 32, y);
    ctx.stroke();
  }

  if (route.length > 1) {
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#44f0d0';
    ctx.beginPath();
    const start = mapPoint(route[0]);
    ctx.moveTo(start.x, start.y);
    for (let i = 1; i < route.length; i++) {
      const point = mapPoint(route[i]);
      ctx.lineTo(point.x, point.y);
    }
    ctx.stroke();

    for (let i = 1; i < route.length - 1; i++) {
      const point = mapPoint(route[i]);
      ctx.fillStyle = `hsl(${(i / Math.max(1, route.length - 2)) * 280 + 20}, 95%, 65%)`;
      ctx.beginPath();
      ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#d9fbff';
      ctx.font = '12px JetBrains Mono, monospace';
      ctx.fillText(String(i), point.x + 8, point.y - 8);
    }
  }

  const startPt = mapPoint(route[0] ?? { x: 0, z: 0 });
  ctx.fillStyle = '#ff8c42';
  ctx.beginPath();
  ctx.arc(startPt.x, startPt.y, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#ffffff';
  ctx.font = '13px JetBrains Mono, monospace';
  ctx.fillText('START', startPt.x + 12, startPt.y - 10);
}

function computeRouteDistance(route: WayPoint[]): number {
  if (route.length < 2) {
    return 0;
  }

  let total = 0;
  for (let i = 1; i < route.length; i++) {
    total += Math.hypot(route[i].x - route[i - 1].x, route[i].z - route[i - 1].z);
  }
  return total;
}

function downloadMissionPdf(): void {
  const pdf = new jsPDF({ orientation: 'portrait', unit: 'pt', format: 'a4' });
  const pageWidth = pdf.internal.pageSize.getWidth();
  const pageHeight = pdf.internal.pageSize.getHeight();
  const margin = 36;

  const paintPdfPage = (): void => {
    pdf.setFillColor(5, 18, 28);
    pdf.rect(0, 0, pageWidth, pageHeight, 'F');
  };

  const drawTableHeader = (yPos: number): void => {
    pdf.setDrawColor(77, 137, 164);
    pdf.setFillColor(10, 32, 46);
    pdf.rect(margin, yPos, pageWidth - margin * 2, 24, 'FD');
    pdf.setTextColor(185, 235, 255);
    pdf.setFontSize(10);
    pdf.text('ID', margin + 8, yPos + 16);
    pdf.text('X', margin + 48, yPos + 16);
    pdf.text('Z', margin + 128, yPos + 16);
    pdf.text('Intensity', margin + 208, yPos + 16);
    pdf.text('Found', margin + 308, yPos + 16);
  };

  paintPdfPage();

  const routePoints = discoveredZones.map((zone) => ({ x: zone.x, y: 8, z: zone.z }));
  const route = computeTspRoute(routePoints);
  const tspDistance = computeRouteDistance(route);

  pdf.setFillColor(8, 28, 40);
  pdf.rect(0, 0, pageWidth, 112, 'F');
  pdf.setTextColor(218, 251, 255);
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(22);
  pdf.text('Mission Report', margin, 48);
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(11);
  pdf.setTextColor(167, 226, 245);
  pdf.text('Autonomous Drone Exploration + TSP Solution', margin, 68);
  pdf.text(`Generated: ${new Date().toLocaleString()}`, margin, 86);

  let y = 132;
  const cardWidth = (pageWidth - margin * 2 - 20) / 3;
  const cards = [
    ['Discovered', `${discoveredZones.length}`],
    ['Distance', `${droneState.distanceTraveled.toFixed(1)} m`],
    ['TSP Length', `${tspDistance.toFixed(1)} units`],
  ];

  cards.forEach((card, idx) => {
    const x = margin + idx * (cardWidth + 10);
    pdf.setFillColor(15, 43, 58);
    pdf.roundedRect(x, y, cardWidth, 58, 8, 8, 'F');
    pdf.setTextColor(142, 212, 239);
    pdf.setFontSize(10);
    pdf.text(card[0], x + 10, y + 18);
    pdf.setTextColor(217, 251, 255);
    pdf.setFontSize(14);
    pdf.setFont('helvetica', 'bold');
    pdf.text(card[1], x + 10, y + 40);
    pdf.setFont('helvetica', 'normal');
  });

  y += 84;
  pdf.setTextColor(217, 251, 255);
  pdf.setFontSize(13);
  pdf.setFont('helvetica', 'bold');
  pdf.text('Explored Points', margin, y);
  y += 16;

  drawTableHeader(y);
  y += 24;

  discoveredZones.forEach((zone, idx) => {
    if (y > pageHeight - 180) {
      pdf.addPage();
      paintPdfPage();
      y = 54;
      drawTableHeader(y);
      y += 24;
    }

    if (idx % 2 === 0) {
      pdf.setFillColor(9, 24, 35);
      pdf.rect(margin, y, pageWidth - margin * 2, 20, 'F');
    }

    pdf.setTextColor(220, 247, 255);
    pdf.setFontSize(10);
    pdf.text(String(zone.id), margin + 8, y + 14);
    pdf.text(zone.x.toFixed(2), margin + 48, y + 14);
    pdf.text(zone.z.toFixed(2), margin + 128, y + 14);
    const reportIntensity = zone.originalIntensity ?? zone.intensity;
    pdf.text(reportIntensity.toFixed(2), margin + 208, y + 14);
    pdf.text(zone.found ? 'YES' : 'NO', margin + 308, y + 14);
    y += 20;
  });

  if (y > pageHeight - 320) {
    pdf.addPage();
    paintPdfPage();
    y = 54;
  }

  y += 10;
  pdf.setTextColor(217, 251, 255);
  pdf.setFontSize(13);
  pdf.setFont('helvetica', 'bold');
  pdf.text('TSP Solution Graph', margin, y);
  y += 12;

  const graphCanvas = document.createElement('canvas');
  graphCanvas.width = 920;
  graphCanvas.height = 620;
  const previousZoom = tspGraphZoom;
  tspGraphZoom = 1;
  renderTspCanvas(graphCanvas, route, discoveredZones);
  tspGraphZoom = previousZoom;

  const imageData = graphCanvas.toDataURL('image/png');
  const imageWidth = pageWidth - margin * 2;
  const imageHeight = (graphCanvas.height / graphCanvas.width) * imageWidth;
  pdf.addImage(imageData, 'PNG', margin, y + 12, imageWidth, imageHeight);

  const filename = `exploration-report-${Date.now()}.pdf`;
  pdf.save(filename);
}

function showTspPopup(includeCurrentPosition = false): void {
  const tspZones = [...discoveredZones];
  if (includeCurrentPosition) {
    tspZones.push({
      id: 0,
      x: droneState.position.x,
      z: droneState.position.z,
      intensity: 0,
      smell: 0,
      originalIntensity: 0,
      originalSmell: 0,
      found: true,
    });
  }

  activeTspPoints = tspZones;
  const routePoints = tspZones.map((zone) => ({ x: zone.x, y: 8, z: zone.z }));
  tspRoute = computeTspRoute(routePoints);
  drawTspGraph(tspRoute, activeTspPoints);
  const popup = getTspPopup();
  popup.classList.remove('hidden');
}

function finalizeMission(reason: string): void {
  if (missionFinalized) {
    return;
  }

  missionFinalized = true;
  missionActive = false;
  droneState.isFlying = false;
  isLaunched = false;
  explorationMode = false;
  droneState.missionComplete = true;
  console.log(`Mission finalized: ${reason}`);
  showTspPopup();
}

async function loadPolicyFromPath(policyPath: string): Promise<void> {
  selectedPolicyPath = policyPath;
  policyStatus = 'loading';
  try {
    const response = policyPath.startsWith('/models/')
      ? await fetch(policyPath, { cache: 'no-cache' })
      : await fetch(`${API_BASE_URL}/api/policies/content?path=${encodeURIComponent(policyPath)}`, { cache: 'no-cache' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = (policyPath.startsWith('/models/')
      ? await response.json()
      : (await response.json() as { path: string; payload: HybridPolicyPayload }).payload) as HybridPolicyPayload;
    if (!Array.isArray(payload.best_actions) || payload.best_actions.length === 0) {
      throw new Error('best_actions missing');
    }

    hybridPolicy = payload;
    policyGridX = payload.config?.grid_x ?? payload.best_actions.length;
    policyGridZ = payload.config?.grid_z ?? payload.best_actions[0]?.length ?? 36;
    policyTBins = payload.config?.t_bins ?? payload.best_actions[0]?.[0]?.length ?? 10;
    policyCoverageBins = payload.config?.coverage_bins ?? payload.best_actions[0]?.[0]?.[0]?.length ?? 8;
    policyStatus = 'ready';
    console.log(`Policy loaded from ${policyPath} (${payload.algorithm ?? 'unknown'})`);
  } catch (err) {
    hybridPolicy = null;
    policyStatus = 'unavailable';
    console.warn(`Policy unavailable for ${policyPath}, using fallback controller.`, err);
  }
}

async function loadHybridPolicy(): Promise<void> {
  await loadPolicyFromPath(selectedPolicyPath);
}

function autoReset(reason: string, fullReset = false): void {
  console.log(`RL RESET: ${reason}`);
  if (fullReset) {
    resetHeatZones(heatZoneCount);
    getTspPopup().classList.add('hidden');
  }
  droneState.position.set(0, 8, 0);
  droneState.velocity.set(0, 0, 0);
  if (fullReset) {
    droneState.distanceTraveled = 0;
  }
  droneState.isFlying = true;
  droneState.missionComplete = false;
  explorationMode = true;
  isLaunched = true;
  if (fullReset) {
    droneTrail = [];
  }
  cumulativeReward = 0;
  episodeSteps = 0;
  lastCollision = false;
  lastThermalSignal = 0;
  currentThermalSignal = 0;
  if (fullReset) {
    coverageCells = new Set<string>();
    recentGridCells = [];
    cellVisitCounts = new Map<string, number>();
  }
  sameActionStreak = 0;
  lastPolicyActionIdx = -1;
  lowMotionStreak = 0;
  loopEscapeTicks = 0;
  collisionRecoveryTimer = 0;
  heldAction = { dvx: 0, dvz: 0 };
  actionDecisionTimer = 0;
  smoothedAction.set(0, 0);
  missionTimeRemaining = MISSION_DURATION_SECONDS;
  missionActive = true;
  missionFinalized = false;
}

function epsilonGreedyAction(observation: DroneObservation): RLAction {
  markCoverageCell();

  if (policyStatus === 'ready' && hybridPolicy) {
    const policyExplore = Math.max(0.015, epsilon * 0.25);
    if (Math.random() >= policyExplore) {
      const actionIdx = getPolicyActionIndex();
      if (actionIdx !== null) {
        if (isLoopingPattern(actionIdx)) {
          loopEscapeTicks = LOOP_ESCAPE_STEPS;
        }

        if (loopEscapeTicks > 0) {
          loopEscapeTicks--;
          const escapeActionIdx = pickEscapeActionIndex(actionIdx);
          return actionIndexToVelocityDelta(escapeActionIdx, observation);
        }

        return actionIndexToVelocityDelta(actionIdx, observation);
      }
    }
  }

  if (Math.random() < epsilon) {
    return {
      dvx: (Math.random() * 2 - 1) * 0.8,
      dvz: (Math.random() * 2 - 1) * 0.8,
    };
  }

  const strongestReading = thermalReadings.length > 0
    ? thermalReadings.reduce((best, curr) => (curr.intensity > best.intensity ? curr : best), thermalReadings[0])
    : null;

  if (!strongestReading) {
    return {
      dvx: -observation.relativePosition.x * 0.25,
      dvz: -observation.relativePosition.z * 0.25,
    };
  }

  const dx = strongestReading.x - droneState.position.x;
  const dz = strongestReading.z - droneState.position.z;
  const mag = Math.max(0.001, Math.hypot(dx, dz));
  const directionScale = Math.max(0.25, 0.65 + observation.thermalGradient * 0.4);

  return {
    dvx: (dx / mag) * directionScale,
    dvz: (dz / mag) * directionScale,
  };
}

function step(action: RLAction, dt: number): { reward: number; done: boolean; observation: DroneObservation } {
  lastThermalSignal = currentThermalSignal;
  const previousDiscovered = discoveredZones.length;
  currentThermalSignal = performThermalScan();

  smoothedAction.lerp(new THREE.Vector2(action.dvx, action.dvz), ACTION_SMOOTHING);
  const dv = new THREE.Vector3(smoothedAction.x, 0, smoothedAction.y).multiplyScalar(droneState.acceleration * dt * 10);
  const desiredVelocity = droneState.velocity.clone().add(dv);

  if (desiredVelocity.length() > droneState.maxSpeed) {
    desiredVelocity.normalize().multiplyScalar(droneState.maxSpeed);
  }

  const avoidance = computeBuildingAvoidance(droneState.position).multiplyScalar(0.08);
  desiredVelocity.add(avoidance);
  if (desiredVelocity.length() > droneState.maxSpeed) {
    desiredVelocity.normalize().multiplyScalar(droneState.maxSpeed);
  }

  const nextPosition = droneState.position.clone().add(desiredVelocity.clone().multiplyScalar(dt));
  const outsideBounds = Math.abs(nextPosition.x) > DRONE_BOUNDARY_LIMIT || Math.abs(nextPosition.z) > DRONE_BOUNDARY_LIMIT;
  if (outsideBounds) {
    nextPosition.x = Math.max(-DRONE_BOUNDARY_LIMIT, Math.min(DRONE_BOUNDARY_LIMIT, nextPosition.x));
    nextPosition.z = Math.max(-DRONE_BOUNDARY_LIMIT, Math.min(DRONE_BOUNDARY_LIMIT, nextPosition.z));
  }
  lastCollision = isPointBlockedByBuilding(nextPosition.x, nextPosition.z, 5.5);

  if (!lastCollision) {
    if (outsideBounds) {
      // Softly damp velocity at the boundary to keep the drone inside without reset loops.
      desiredVelocity.multiplyScalar(0.68);
    }
    droneState.velocity.copy(desiredVelocity);
    if (droneState.velocity.length() < VELOCITY_DEADZONE) {
      droneState.velocity.set(0, 0, 0);
    }
    droneState.position.copy(nextPosition);
    droneState.distanceTraveled += droneState.velocity.length() * dt;
    if (desiredVelocity.length() < 0.22) {
      lowMotionStreak++;
    } else {
      lowMotionStreak = Math.max(0, lowMotionStreak - 1);
    }
  } else {
    droneState.velocity.multiplyScalar(0.2);
    lowMotionStreak += 2;
  }

  let reward = 0;
  reward += (currentThermalSignal - lastThermalSignal) * 8;
  reward -= 0.012;
  if (discoveredZones.length > previousDiscovered) reward += 2.4;
  if (lastCollision) reward -= outsideBounds ? 8 : 20;

  cumulativeReward += reward;
  episodeSteps++;
  epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);

  const done = cumulativeReward < REWARD_RESET_THRESHOLD;
  const observation = getDroneObservation();

  return { reward, done, observation };
}

// ML-based autonomous drone movement
function updateDroneMovement(dt: number): void {
  if (!droneState.isFlying) return;

  collisionRecoveryTimer = Math.max(0, collisionRecoveryTimer - dt);
  actionDecisionTimer = Math.max(0, actionDecisionTimer - dt);

  const observation = getDroneObservation();
  if (actionDecisionTimer <= 0 || lastCollision) {
    heldAction = epsilonGreedyAction(observation);
    actionDecisionTimer = ACTION_DECISION_INTERVAL;
  }
  const action = heldAction;
  step(action, dt);

  if (lastCollision && collisionRecoveryTimer <= 0) {
    collisionRecoveryTimer = COLLISION_RECOVERY_COOLDOWN;
    const inward = new THREE.Vector3(-droneState.position.x, 0, -droneState.position.z).normalize().multiplyScalar(2.5);
    droneState.velocity.add(inward);
  }

  // Update drone visual
  droneGroup.position.copy(droneState.position);

  // Rotate drone towards movement direction
  if (droneState.velocity.length() > 0.1) {
    const lookTarget = droneState.position.clone().add(droneState.velocity);
    droneGroup.lookAt(lookTarget);
  }

  // Spin propellers
  propellers.forEach((prop, idx) => {
    prop.rotation.z += 0.72 + idx * 0.03;
  });

  droneRing.rotation.z += 0.015;

  updateDroneTrail(dt);
}

// UI Setup
let setupDone = false;

function setupUI(): void {
  if (setupDone) return;
  setupDone = true;

  controls.innerHTML = `
    <div class="row">
      <div class="label">Autonomous Drone</div>
      <div class="value" id="drone-status">Ready</div>
    </div>
    <div class="row">
      <div class="label">Search Countdown</div>
      <div class="value" id="mission-countdown">05:00</div>
    </div>
    <div class="row top-gap" style="border-top: 1px solid #0a3a4a; padding-top: 10px;">
      <div class="label">Policy</div>
      <div class="value" id="policy-source-label">Hybrid DRL</div>
    </div>
    <div style="display: flex; gap: 5px; margin-top: 8px;">
      <select id="exploration-policy-select" style="width: 100%; padding: 5px; background: #0a1a2a; color: #00ffdd; border: 1px solid #00ffdd;">
        <option value="trainer/models/dyna_q_policy.json">Dyna-Q (default)</option>
        <option value="/models/hybrid_drl_explorer_policy.json" selected>Hybrid DRL</option>
        <option value="trainer/models/vanilla_q_policy.json">Vanilla Q</option>
      </select>
    </div>
    <button id="launch-btn">Start Exploration</button>
    <button id="stop-btn">Emergency Stop</button>

    <div class="row top-gap" style="border-top: 1px solid #0a3a4a; padding-top: 10px;">
      <div class="label">DRONE SPEED</div>
      <div class="value" id="speed-val">15.0 m/s</div>
    </div>
    <div style="display: flex; gap: 5px; margin-top: 8px;">
      <input type="number" id="speed-input" min="1" max="40" step="0.5" value="15" style="width: 50%; padding: 5px; background: #0a1a2a; color: #00ffdd; border: 1px solid #00ffdd;">
      <button id="apply-speed-btn" style="width: 50%;">Apply Speed</button>
    </div>

    <div class="row top-gap" style="border-top: 1px solid #0a3a4a; padding-top: 10px;">
      <div class="label">HEAT ZONES</div>
      <div class="value" id="heat-zone-count-val">3</div>
    </div>
    <div style="display: flex; gap: 5px; margin-top: 8px;">
      <input type="number" id="heat-zone-count" min="1" max="25" value="3" style="width: 50%; padding: 5px; background: #0a1a2a; color: #00ffdd; border: 1px solid #00ffdd;">
      <button id="apply-zones-btn" style="width: 50%;">Apply Zones</button>
    </div>
    <button id="add-zone-btn-quick" style="width: 100%; margin-top: 5px;">Add 1 Random Heat Zone</button>
    
    <div class="row top-gap" style="border-top: 1px solid #0a3a4a; padding-top: 10px;">
      <div class="label">CAMERA VIEWS</div>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px; margin-top: 8px;">
      <button id="cam-topdown" style="padding: 8px;">↓ Top-Down</button>
      <button id="cam-side" style="padding: 8px;">⊢ Side</button>
      <button id="cam-follow" style="padding: 8px;">⟲ Follow</button>
      <button id="cam-isometric" style="padding: 8px;">◈ Isometric</button>
      <button id="cam-freelook" style="padding: 8px; grid-column: 1 / -1;">⟲ Free-Look</button>
    </div>
    
    <div class="row top-gap">
      <div class="label">Zones Discovered</div>
      <div class="value" id="zones-discovered">0</div>
    </div>
    <div class="row">
      <div class="label">Zones Visited</div>
      <div class="value" id="zones-visited">0</div>
    </div>
    <div class="row">

      <div class="label">Distance</div>
      <div class="value" id="distance-val">0.0 m</div>
    </div>
    <button id="tsp-current-btn">Current TSP (Explored + Drone)</button>
    <button id="export-btn">Export Mission Report</button>
  `;

  const launchBtn = document.getElementById('launch-btn');
  const stopBtn = document.getElementById('stop-btn');
  const tspCurrentBtn = document.getElementById('tsp-current-btn');
  const exportBtn = document.getElementById('export-btn');
  const applySpeedBtn = document.getElementById('apply-speed-btn');
  const applyZonesBtn = document.getElementById('apply-zones-btn');
  const addZoneQuickBtn = document.getElementById('add-zone-btn-quick');
  const heatZoneCountInput = document.getElementById('heat-zone-count') as HTMLInputElement | null;
  const heatZoneCountVal = document.getElementById('heat-zone-count-val') as HTMLDivElement | null;
  const speedInput = document.getElementById('speed-input') as HTMLInputElement | null;
  const speedVal = document.getElementById('speed-val') as HTMLDivElement | null;
  const camTopBtn = document.getElementById('cam-topdown');
  const camSideBtn = document.getElementById('cam-side');
  const camFollowBtn = document.getElementById('cam-follow');
  const camIsometricBtn = document.getElementById('cam-isometric');
  const camFreelookBtn = document.getElementById('cam-freelook');
  const explorationPolicySelect = document.getElementById('exploration-policy-select') as HTMLSelectElement | null;
  const policySourceLabel = document.getElementById('policy-source-label') as HTMLDivElement | null;

  droneStatusEl = document.getElementById('drone-status');
  missionCountdownEl = document.getElementById('mission-countdown');
  zonesDiscoveredEl = document.getElementById('zones-discovered');
  zonesVisitedEl = document.getElementById('zones-visited');
  distanceValEl = document.getElementById('distance-val');
  speedValEl = document.getElementById('speed-val');

  function updateCameraButtonStyles(): void {
    [camTopBtn, camSideBtn, camFollowBtn, camIsometricBtn, camFreelookBtn].forEach((btn) => {
      btn?.classList.remove('active');
    });
    if (currentCameraView === 'topdown') camTopBtn?.classList.add('active');
    else if (currentCameraView === 'side') camSideBtn?.classList.add('active');
    else if (currentCameraView === 'follow') camFollowBtn?.classList.add('active');
    else if (currentCameraView === 'isometric') camIsometricBtn?.classList.add('active');
    else if (currentCameraView === 'freelook') camFreelookBtn?.classList.add('active');
  }

  camTopBtn?.addEventListener('click', () => {
    currentCameraView = 'topdown';
    updateCameraButtonStyles();
  });
  camSideBtn?.addEventListener('click', () => {
    currentCameraView = 'side';
    updateCameraButtonStyles();
  });
  camFollowBtn?.addEventListener('click', () => {
    currentCameraView = 'follow';
    updateCameraButtonStyles();
  });
  camIsometricBtn?.addEventListener('click', () => {
    currentCameraView = 'isometric';
    updateCameraButtonStyles();
  });
  camFreelookBtn?.addEventListener('click', () => {
    currentCameraView = 'freelook';
    updateCameraButtonStyles();
  });

  updateCameraButtonStyles();

  const policyChoices = [
    { label: 'Dyna-Q (default)', path: 'trainer/models/dyna_q_policy.json' },
    { label: 'Hybrid DRL', path: '/models/hybrid_drl_explorer_policy.json' },
    { label: 'Vanilla Q', path: 'trainer/models/vanilla_q_policy.json' },
  ];

  async function refreshExplorationPolicyOptions(): Promise<void> {
    if (!explorationPolicySelect) return;
    const currentValue = explorationPolicySelect.value;
    explorationPolicySelect.innerHTML = '';

    policyChoices.forEach((choice) => {
      const option = document.createElement('option');
      option.value = choice.path;
      option.textContent = choice.label;
      explorationPolicySelect.appendChild(option);
    });

    try {
      const response = await fetch(`${API_BASE_URL}/api/policies`);
      const policies = await response.json() as string[];
      policies
        .filter((policyPath) => policyPath.includes('local_runs'))
        .forEach((policyPath) => {
          if (policyChoices.some((choice) => choice.path === policyPath)) return;
          const option = document.createElement('option');
          option.value = policyPath;
          option.textContent = `Local trained: ${policyPath.split(/[\\\/]/).pop() || policyPath}`;
          explorationPolicySelect.appendChild(option);
        });
    } catch (err) {
      console.warn('Could not load local policy list', err);
    }

    if (currentValue && Array.from(explorationPolicySelect.options).some((option) => option.value === currentValue)) {
      explorationPolicySelect.value = currentValue;
    } else {
      explorationPolicySelect.value = selectedPolicyPath;
    }

    const selectedOption = explorationPolicySelect.selectedOptions[0];
    if (policySourceLabel && selectedOption) {
      policySourceLabel.textContent = selectedOption.textContent || 'Policy';
    }
  }

  void refreshExplorationPolicyOptions();

  explorationPolicySelect?.addEventListener('change', () => {
    const selectedOption = explorationPolicySelect.selectedOptions[0];
    if (policySourceLabel && selectedOption) {
      policySourceLabel.textContent = selectedOption.textContent || 'Policy';
    }
  });

  if (launchBtn) {
    launchBtn.addEventListener('click', async () => {
      console.log('Launch button clicked; isLaunched=', isLaunched);
      if (!isLaunched) {
        console.log('Starting RL exploration episode...');
        try {
          const policyPath = explorationPolicySelect?.value || selectedPolicyPath;
          await loadPolicyFromPath(policyPath);
          if (policyStatus !== 'ready') {
            alert(`Failed to load policy: ${policyPath}`);
            return;
          }
          autoReset('manual-start', true);
        } catch (e) {
          console.error('autoReset error:', e);
          alert('Failed to start exploration (see console)');
        }
      }
    });
  } else {
    console.warn('Launch button not found when setting up UI');
  }

  stopBtn?.addEventListener('click', () => {
    droneState.isFlying = false;
    isLaunched = false;
    explorationMode = false;
    missionActive = false;
  });

  applySpeedBtn?.addEventListener('click', () => {
    if (!speedInput) return;
    const requestedSpeed = Math.min(MAX_DRONE_SPEED, Math.max(1, parseFloat(speedInput.value) || droneState.maxSpeed));
    speedInput.value = requestedSpeed.toFixed(1);
    droneState.maxSpeed = requestedSpeed;
    if (speedVal) speedVal.textContent = `${requestedSpeed.toFixed(1)} m/s`;
    console.log(`Drone speed set to ${requestedSpeed.toFixed(1)} m/s`);
  });

  applyZonesBtn?.addEventListener('click', () => {
    if (!heatZoneCountInput || !heatZoneCountVal) return;
    const requestedCount = Math.min(25, Math.max(1, parseInt(heatZoneCountInput.value, 10) || 3));
    heatZoneCountInput.value = String(requestedCount);
    heatZoneCountVal.textContent = String(requestedCount);
    resetHeatZones(requestedCount);
    console.log(`Heat zones regenerated: ${requestedCount}`);
  });

  addZoneQuickBtn?.addEventListener('click', () => {
    const requestedCount = Math.min(25, heatZoneCount + 1);
    if (heatZoneCountInput) heatZoneCountInput.value = String(requestedCount);
    if (heatZoneCountVal) heatZoneCountVal.textContent = String(requestedCount);
    resetHeatZones(requestedCount);
    console.log(`Heat zones increased to ${requestedCount}`);
  });

  tspCurrentBtn?.addEventListener('click', () => {
    showTspPopup(true);
  });

  exportBtn?.addEventListener('click', () => {
    downloadMissionPdf();
  });

  // --- Thermal mapping controls (bin size and offset) ---
  // Load saved mapping if present
  try {
    const saved = localStorage.getItem('thermal_mapping');
    if (saved) {
      const parsed = JSON.parse(saved);
      if (typeof parsed.binSize === 'number') thermalBinSize = parsed.binSize;
      if (typeof parsed.offset === 'number') thermalOffset = parsed.offset;
      if (typeof parsed.policyTBins === 'number') policyTBins = parsed.policyTBins;
    }
  } catch (e) {
    console.warn('Failed to load thermal_mapping from localStorage', e);
  }

  const thermalPanel = document.createElement('div');
  thermalPanel.style.cssText = 'margin-top:12px; padding:8px; background: rgba(5,12,18,0.6); border: 1px dashed #00606f; border-radius:6px; font-size:12px; color:#bcefff;';
  thermalPanel.innerHTML = `
    <div style="color:#00ffcc; font-weight:bold; margin-bottom:6px;">THERMAL MAPPING</div>
    <div style="display:flex; gap:6px; align-items:center; margin-bottom:6px;">
      <label style="min-width:90px;">Bin Size (°C):</label>
      <input id="thermal-bin-size" type="number" min="1" max="200" step="1" style="flex:1; background:#05202b; color:#00ffcc; border:1px solid #00ffcc; padding:4px;" value="${thermalBinSize}">
    </div>
    <div style="display:flex; gap:6px; align-items:center; margin-bottom:6px;">
      <label style="min-width:90px;">Offset (°C):</label>
      <input id="thermal-offset" type="number" step="0.1" style="flex:1; background:#05202b; color:#00ffcc; border:1px solid #00ffcc; padding:4px;" value="${thermalOffset}">
    </div>
    <div style="display:flex; gap:6px; align-items:center;">
      <label style="min-width:90px;">Policy T-Bins:</label>
      <input id="policy-t-bins" type="number" min="1" max="200" step="1" style="width:80px; background:#05202b; color:#00ffcc; border:1px solid #00ffcc; padding:4px;" value="${policyTBins}">
      <div style="flex:1; color:#89f7ff; font-size:11px; text-align:right;">Controls adjust binning for thermal->policy mapping</div>
    </div>
  `;

  controls.appendChild(thermalPanel);

  const thermalBinSizeInput = document.getElementById('thermal-bin-size') as HTMLInputElement | null;
  const thermalOffsetInput = document.getElementById('thermal-offset') as HTMLInputElement | null;
  const policyTBinsInput = document.getElementById('policy-t-bins') as HTMLInputElement | null;

  function persistThermalMapping(): void {
    try {
      localStorage.setItem('thermal_mapping', JSON.stringify({ binSize: thermalBinSize, offset: thermalOffset, policyTBins }));
    } catch (e) {
      console.warn('Failed to persist thermal mapping', e);
    }
  }

  thermalBinSizeInput?.addEventListener('change', () => {
    const v = Math.max(1, Math.floor(Number(thermalBinSizeInput.value) || 12));
    thermalBinSize = v;
    thermalBinSizeInput.value = String(v);
    persistThermalMapping();
    console.log('thermalBinSize set to', thermalBinSize);
  });

  thermalOffsetInput?.addEventListener('change', () => {
    const v = Number(thermalOffsetInput.value) || 0;
    thermalOffset = v;
    thermalOffsetInput.value = String(v);
    persistThermalMapping();
    console.log('thermalOffset set to', thermalOffset);
  });

  policyTBinsInput?.addEventListener('change', () => {
    const v = Math.max(1, Math.floor(Number(policyTBinsInput.value) || policyTBins));
    policyTBins = v;
    policyTBinsInput.value = String(v);
    persistThermalMapping();
    console.log('policyTBins overridden to', policyTBins);
  });


  // --- Environment Customization Panel ---
  const envPanel = document.createElement('div');
  // Make panel scrollable without overlapping other controls
  envPanel.style.cssText = 'position: relative; margin-top: 12px; padding: 10px; background: rgba(10, 26, 42, 0.88); border: 1px solid #00ffcc; border-radius: 7px; max-height: 52vh; overflow-y: auto; overflow-x: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.45); scrollbar-gutter: stable;';
  
  const envTitle = document.createElement('div');
  envTitle.style.cssText = 'color: #00ffcc; font-weight: bold; margin-bottom: 10px; font-size: 14px;';
  envTitle.textContent = 'ENVIRONMENT BUILDER - CLICK TO PLACE';
  envPanel.appendChild(envTitle);

  // Instructions
  const instrDiv = document.createElement('div');
  instrDiv.style.cssText = 'color: #bcefff; font-size: 11px; margin-bottom: 8px; padding: 6px; background: rgba(0,0,0,0.3); border-radius: 4px; border-left: 2px solid #00ffcc;';
  instrDiv.textContent = '1. Select object type  2. Adjust height  3. Click on 3D canvas to place  4. Save when done';
  envPanel.appendChild(instrDiv);

  // Object type selector
  const typeRow = document.createElement('div');
  typeRow.style.cssText = 'margin-bottom: 8px; display: flex; gap: 6px; flex-wrap: wrap;';
  
  const typeLabel = document.createElement('span');
  typeLabel.textContent = 'Add:';
  typeLabel.style.cssText = 'color: #bcefff; font-size: 12px; align-self: center;';
  typeRow.appendChild(typeLabel);

  const buildingBtn = document.createElement('button');
  buildingBtn.textContent = 'Building';
  buildingBtn.style.cssText = 'padding: 6px 10px; font-size: 11px; background: #003a4a; color: #00ffcc; border: 1px solid #00ffcc; border-radius: 4px; cursor: pointer; font-weight: bold;';
  typeRow.appendChild(buildingBtn);

  const machineBtn = document.createElement('button');
  machineBtn.textContent = 'Machine';
  machineBtn.style.cssText = 'padding: 6px 10px; font-size: 11px; background: #1a3a4a; color: #00ffff; border: 1px solid #00ffff; border-radius: 4px; cursor: pointer;';
  typeRow.appendChild(machineBtn);

  const pipeBtn = document.createElement('button');
  pipeBtn.textContent = 'Pipe';
  pipeBtn.style.cssText = 'padding: 6px 10px; font-size: 11px; background: #1a3a4a; color: #ffaa00; border: 1px solid #ffaa00; border-radius: 4px; cursor: pointer;';
  typeRow.appendChild(pipeBtn);

  envPanel.appendChild(typeRow);

  // Height control
  const heightRow = document.createElement('div');
  heightRow.style.cssText = 'margin-bottom: 8px; display: flex; gap: 6px; align-items: center;';
  
  const heightLabel = document.createElement('span');
  heightLabel.textContent = 'Height:';
  heightLabel.style.cssText = 'color: #bcefff; font-size: 12px; min-width: 50px;';
  heightRow.appendChild(heightLabel);

  const heightInput = document.createElement('input');
  heightInput.type = 'range';
  heightInput.min = '1';
  heightInput.max = '50';
  heightInput.value = '10';
  heightInput.style.cssText = 'flex: 1; cursor: pointer;';
  heightRow.appendChild(heightInput);

  const heightVal = document.createElement('span');
  heightVal.textContent = '10 units';
  heightVal.style.cssText = 'color: #bcefff; font-size: 12px; min-width: 70px; text-align: right;';
  heightRow.appendChild(heightVal);

  envPanel.appendChild(heightRow);

  // Coordinates display
  const coordDiv = document.createElement('div');
  coordDiv.style.cssText = 'margin-bottom: 8px; padding: 6px; background: rgba(0,0,0,0.3); border-radius: 4px; font-size: 11px; color: #00ffcc;';
  coordDiv.innerHTML = '<strong>Mouse Coords:</strong> X: 0.0, Z: 0.0 | <strong>Mode:</strong> <span style="color: #00ffcc;">Building</span>';
  envPanel.appendChild(coordDiv);

  // Environment objects list
  const objListLabel = document.createElement('div');
  objListLabel.textContent = 'Custom Objects:';
  objListLabel.style.cssText = 'color: #00ffcc; font-size: 12px; margin-top: 10px; margin-bottom: 6px;';
  envPanel.appendChild(objListLabel);

  const objList = document.createElement('div');
  objList.style.cssText = 'max-height: 100px; overflow-y: auto; background: rgba(0, 0, 0, 0.5); padding: 6px; border-radius: 4px; font-size: 11px; color: #bcefff;';
  objList.innerHTML = 'No objects added';
  envPanel.appendChild(objList);

  // Save/Load buttons
  const saveLoadRow = document.createElement('div');
  saveLoadRow.style.cssText = 'margin-top: 10px; display: flex; gap: 6px; justify-content: space-between;';

  const clearBtn = document.createElement('button');
  clearBtn.textContent = 'Clear All';
  clearBtn.style.cssText = 'flex: 1; padding: 8px; background: #3a1a1a; color: #ff6666; border: 1px solid #ff6666; border-radius: 4px; cursor: pointer; font-size: 11px;';
  saveLoadRow.appendChild(clearBtn);

  const clearBuildingsBtn = document.createElement('button');
  clearBuildingsBtn.textContent = 'Clear Buildings';
  clearBuildingsBtn.style.cssText = 'flex: 1; padding: 8px; background: #2a2a1a; color: #ffcf7a; border: 1px solid #ffcf7a; border-radius: 4px; cursor: pointer; font-size: 11px;';
  saveLoadRow.appendChild(clearBuildingsBtn);

  const saveBtn = document.createElement('button');
  saveBtn.textContent = 'Save Environment';
  saveBtn.style.cssText = 'flex: 1; padding: 8px; background: #1a3a2a; color: #00ffaa; border: 1px solid #00ffaa; border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: bold;';
  saveLoadRow.appendChild(saveBtn);

  const loadBtn = document.createElement('button');
  loadBtn.textContent = 'Load Saved';
  loadBtn.style.cssText = 'flex: 1; padding: 8px; background: #1a2a3a; color: #00aaff; border: 1px solid #00aaff; border-radius: 4px; cursor: pointer; font-size: 11px;';
  saveLoadRow.appendChild(loadBtn);

  envPanel.appendChild(saveLoadRow);

  const trainingTitle = document.createElement('div');
  trainingTitle.style.cssText = 'color: #ffcf7a; font-weight: bold; margin: 14px 0 8px; font-size: 13px; border-top: 1px solid rgba(255,207,122,0.35); padding-top: 10px;';
  trainingTitle.textContent = 'TRAINING LAUNCHER';
  envPanel.appendChild(trainingTitle);

  // Policy selector (available policies + current)
  const policyRow = document.createElement('div');
  policyRow.style.cssText = 'display:flex; gap:6px; align-items:center; margin-bottom:8px;';
  const policyLabel = document.createElement('span');
  policyLabel.textContent = 'Policy:';
  policyLabel.style.cssText = 'color:#bcefff; font-size:12px; min-width:48px;';
  policyRow.appendChild(policyLabel);
  const policySelect = document.createElement('select');
  policySelect.style.cssText = 'flex:1; min-width:180px; padding:6px; background:#05202b; color:#00ffcc; border:1px solid #00ffcc; border-radius:4px;';
  policyRow.appendChild(policySelect);
  const activateBeforeBtn = document.createElement('button');
  activateBeforeBtn.textContent = 'Activate before run';
  activateBeforeBtn.style.cssText = 'padding:6px 8px; font-size:11px; background:#003a4a; color:#00ffcc; border:1px solid #00ffcc; border-radius:4px; cursor:pointer;';
  policyRow.appendChild(activateBeforeBtn);
  envPanel.appendChild(policyRow);

  // Load policies from server and add predefined policy options
  async function refreshPolicies() {
    try {
      const res = await fetch(`${API_BASE_URL}/api/policies`);
      const list: string[] = await res.json();
      policySelect.innerHTML = '';

      // Predefined policy choices (will appear first)
      const predefined = [
        { label: 'Dyna-Q (default)', path: 'trainer/models/dyna_q_policy.json' },
        { label: 'Hybrid DRL (hybrid_drl_explorer)', path: '/models/hybrid_drl_explorer_policy.json' },
        { label: 'Vanilla Q (vanilla_q_policy)', path: 'trainer/models/vanilla_q_policy.json' },
      ];
      predefined.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.path;
        opt.textContent = p.label;
        policySelect.appendChild(opt);
      });

      // add current marker (if different)
      try {
        const curRes = await fetch(`${API_BASE_URL}/api/policy/current`);
        const cur = await curRes.json();
        if (cur.exists && cur.path) {
          const curPath = cur.path as string;
          // avoid duplicate
          if (!predefined.find(p => p.path === curPath)) {
            const opt = document.createElement('option');
            opt.value = curPath;
            opt.textContent = `Default (current): ${curPath.split('/').pop() || curPath}`;
            policySelect.appendChild(opt);
          }
        }
      } catch (err) {
        // ignore
      }

      // add discovered policies (local runs, backups)
      list.forEach(p => {
        // skip duplicates
        const exists = Array.from(policySelect.options).some(o => o.value === p);
        if (exists) return;
        const opt = document.createElement('option');
        opt.value = p;
        opt.textContent = p.split(/[\\\/]/).pop() || p;
        policySelect.appendChild(opt);
      });

      // try to select latest local_runs if present
      const localOpt = list.find(p => p.includes('local_runs'));
      if (localOpt) policySelect.value = localOpt;
    } catch (e) {
      console.warn('Could not refresh policies', e);
    }
  }
  void refreshPolicies();

  activateBeforeBtn.addEventListener('click', async () => {
    const path = policySelect.value;
    if (!path) return;
    try {
      const r = await fetch(`${API_BASE_URL}/api/policies/activate`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ path }) });
      const j = await r.json();
      if (j.ok) {
        trainingStatus.textContent = `Activated policy: ${path.split(/[\\\/]/).pop()}`;
      } else {
        trainingStatus.textContent = `Failed to activate policy: ${JSON.stringify(j)}`;
      }
    } catch (err) {
      trainingStatus.textContent = `Activate error: ${String(err)}`;
    }
  });

  const presetRow = document.createElement('div');
  presetRow.style.cssText = 'display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin-bottom: 8px;';
  const presetLabel = document.createElement('span');
  presetLabel.textContent = 'Preset:';
  presetLabel.style.cssText = 'color: #bcefff; font-size: 12px; min-width: 48px;';
  presetRow.appendChild(presetLabel);
  const presetSelect = document.createElement('select');
  presetSelect.style.cssText = 'flex: 1; min-width: 140px; padding: 6px; background: #05202b; color: #00ffcc; border: 1px solid #00ffcc; border-radius: 4px;';
  presetSelect.innerHTML = `
    <option value="current">Current saved environment</option>
    <option value="empty">Empty environment</option>
    <option value="building_cluster">Building cluster</option>
    <option value="industrial_mix">Industrial mix</option>
    <option value="pipe_corridor">Pipe corridor</option>
  `;
  presetRow.appendChild(presetSelect);
  const applyPresetBtn = document.createElement('button');
  applyPresetBtn.textContent = 'Apply';
  applyPresetBtn.style.cssText = 'padding: 6px 10px; font-size: 11px; background: #003a4a; color: #00ffcc; border: 1px solid #00ffcc; border-radius: 4px; cursor: pointer; font-weight: bold;';
  presetRow.appendChild(applyPresetBtn);
  envPanel.appendChild(presetRow);

  const modeRow = document.createElement('div');
  modeRow.style.cssText = 'display:flex; gap:6px; align-items:center; flex-wrap: wrap; margin-bottom: 8px;';
  const modeLabel = document.createElement('span');
  modeLabel.textContent = 'Mode:';
  modeLabel.style.cssText = 'color:#bcefff; font-size:12px; min-width:48px;';
  modeRow.appendChild(modeLabel);
  const trainingModeSelect = document.createElement('select');
  trainingModeSelect.style.cssText = 'flex:1; min-width:140px; padding: 6px; background: #05202b; color: #00ffcc; border: 1px solid #00ffcc; border-radius: 4px;';
  trainingModeSelect.innerHTML = `
    <option value="kaggle">Kaggle Training</option>
    <option value="local">Local Training (train_dyna_q.py)</option>
  `;
  modeRow.appendChild(trainingModeSelect);
  envPanel.appendChild(modeRow);

  const mapRow = document.createElement('div');
  mapRow.style.cssText = 'display:flex; gap:6px; align-items:center; flex-wrap: wrap; margin-bottom: 8px;';
  const mapLabel = document.createElement('span');
  mapLabel.textContent = 'Map:';
  mapLabel.style.cssText = 'color:#bcefff; font-size:12px; min-width:48px;';
  mapRow.appendChild(mapLabel);
  const mapInput = document.createElement('input');
  mapInput.type = 'text';
  mapInput.value = 'sample_map.png';
  mapInput.placeholder = 'sample_map.png';
  mapInput.style.cssText = 'flex:1; min-width: 180px; padding: 6px; background: #05202b; color: #dfeffa; border: 1px solid #00aaff; border-radius: 4px;';
  mapRow.appendChild(mapInput);
  envPanel.appendChild(mapRow);

  const projectRow = document.createElement('div');
  projectRow.style.cssText = 'display:flex; gap:6px; align-items:center; flex-wrap: wrap; margin-bottom: 8px;';
  const projectLabel = document.createElement('span');
  projectLabel.textContent = 'Project:';
  projectLabel.style.cssText = 'color:#bcefff; font-size:12px; min-width:48px;';
  projectRow.appendChild(projectLabel);
  const projectInput = document.createElement('input');
  projectInput.type = 'text';
  projectInput.value = 'project01';
  projectInput.placeholder = 'project01';
  projectInput.style.cssText = 'flex:1; min-width: 180px; padding: 6px; background: #05202b; color: #dfeffa; border: 1px solid #00aaff; border-radius: 4px;';
  projectRow.appendChild(projectInput);
  envPanel.appendChild(projectRow);

  const apiRow = document.createElement('div');
  apiRow.style.cssText = 'display:flex; gap:6px; align-items:center; flex-wrap: wrap; margin-bottom: 8px;';
  const apiLabel = document.createElement('span');
  apiLabel.textContent = 'Key:';
  apiLabel.style.cssText = 'color:#bcefff; font-size:12px; min-width:48px;';
  apiRow.appendChild(apiLabel);
  const useDefaultSelect = document.createElement('select');
  useDefaultSelect.style.cssText = 'padding: 6px; background: #05202b; color: #00ffcc; border: 1px solid #00ffcc; border-radius: 4px;';
  useDefaultSelect.innerHTML = `
    <option value="true">Use repository default</option>
    <option value="false">Use pasted token</option>
  `;
  apiRow.appendChild(useDefaultSelect);
  const tokenInput = document.createElement('input');
  tokenInput.type = 'password';
  tokenInput.placeholder = 'Kaggle API token';
  tokenInput.style.cssText = 'flex:1; min-width: 180px; padding: 6px; background: #05202b; color: #dfeffa; border: 1px solid #00aaff; border-radius: 4px;';
  apiRow.appendChild(tokenInput);
  envPanel.appendChild(apiRow);

  const updateModeUI = () => {
    const localMode = trainingModeSelect.value === 'local';
    mapInput.disabled = localMode;
    tokenInput.disabled = localMode;
    useDefaultSelect.disabled = localMode;
    mapLabel.style.opacity = localMode ? '0.55' : '1';
    apiLabel.style.opacity = localMode ? '0.55' : '1';
  };
  trainingModeSelect.addEventListener('change', updateModeUI);
  updateModeUI();

  const rlRow = document.createElement('div');
  rlRow.style.cssText = 'display:flex; gap:6px; flex-wrap: wrap; margin-bottom: 8px;';
  const makeNumberField = (labelText: string, initialValue: string) => {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex; gap:4px; align-items:center;';
    const label = document.createElement('span');
    label.textContent = labelText;
    label.style.cssText = 'color:#bcefff; font-size:12px;';
    const input = document.createElement('input');
    input.type = 'number';
    input.value = initialValue;
    input.style.cssText = 'width:92px; padding:6px; background:#05202b; color:#dfeffa; border:1px solid #00aaff; border-radius:4px;';
    wrapper.appendChild(label);
    wrapper.appendChild(input);
    rlRow.appendChild(wrapper);
    return input;
  };
  const trainSecondsInput = makeNumberField('Sec', '5400');
  const planningStepsInput = makeNumberField('Plan', '60');
  const episodesInput = makeNumberField('Episodes', '15000');
  envPanel.appendChild(rlRow);

  const launchRow = document.createElement('div');
  launchRow.style.cssText = 'display:flex; gap:6px; align-items:flex-start; flex-wrap: wrap; margin-top: 6px;';
  const trainingLaunchBtn = document.createElement('button');
  trainingLaunchBtn.textContent = 'Start Training';
  trainingLaunchBtn.style.cssText = 'padding: 10px 14px; background: linear-gradient(135deg, #00ffcc, #00aaff); color: #051521; border: none; border-radius: 6px; cursor: pointer; font-weight: bold;';
  launchRow.appendChild(trainingLaunchBtn);
  const trainingStatus = document.createElement('pre');
  trainingStatus.style.cssText = 'flex:1; min-width: 220px; margin: 0; background: rgba(0, 0, 0, 0.38); color: #dfeffa; padding: 8px; border-radius: 6px; font-size: 11px; white-space: pre-wrap;';
  trainingStatus.textContent = 'No training run started yet.';
  launchRow.appendChild(trainingStatus);
  envPanel.appendChild(launchRow);
  controls.appendChild(envPanel);

  // Environment data structure
  let environmentObjects: CustomEnvironmentObject[] = [];
  let selectedObjectType = 'building';
  const cloneEnvironmentObject = (obj: CustomEnvironmentObject): CustomEnvironmentObject => ({
    type: obj.type,
    height: obj.height,
    x: obj.x,
    z: obj.z,
    width: obj.width,
    depth: obj.depth,
  });
  const serializeEnvironmentObjects = (objects: CustomEnvironmentObject[]): CustomEnvironmentObject[] => objects.map(cloneEnvironmentObject);
  const loadSavedEnvironmentObjects = (): CustomEnvironmentObject[] => {
    const saved = localStorage.getItem('default_environment');
    if (!saved) return [];
    try {
      const parsed = JSON.parse(saved);
      if (!Array.isArray(parsed)) return [];
      return parsed.map((obj: CustomEnvironmentObject) => cloneEnvironmentObject(obj));
    } catch {
      return [];
    }
  };
  const buildPresetEnvironmentObjects = (preset: string): CustomEnvironmentObject[] => {
    if (preset === 'empty') return [];
    if (preset === 'building_cluster') {
      return [
        { type: 'building', height: 18, x: -22, z: -8 },
        { type: 'building', height: 24, x: -8, z: 10 },
        { type: 'building', height: 15, x: 12, z: -6 },
        { type: 'machine', height: 10, x: 20, z: 14 },
      ];
    }
    if (preset === 'industrial_mix') {
      return [
        { type: 'building', height: 20, x: -26, z: -18 },
        { type: 'machine', height: 12, x: -6, z: -10 },
        { type: 'pipe', height: 14, x: 9, z: 6 },
        { type: 'building', height: 16, x: 24, z: 18 },
        { type: 'pipe', height: 12, x: -14, z: 20 },
      ];
    }
    if (preset === 'pipe_corridor') {
      return [
        { type: 'pipe', height: 18, x: -20, z: -20 },
        { type: 'pipe', height: 18, x: -6, z: -6 },
        { type: 'pipe', height: 18, x: 8, z: 8 },
        { type: 'building', height: 14, x: 18, z: 18 },
      ];
    }
    return [
      { type: 'building', height: 18, x: -18, z: -12 },
      { type: 'machine', height: 10, x: 8, z: 6 },
      { type: 'pipe', height: 12, x: 18, z: -4 },
    ];
  };
  const getStableBuildingFootprint = (x: number, z: number, height: number): { width: number; depth: number } => {
    const seed = Math.sin(x * 12.9898 + z * 78.233 + height * 37.719) * 43758.5453;
    const frac = seed - Math.floor(seed);
    const secondSeed = Math.sin(x * 93.9898 + z * 67.345 + height * 11.113) * 24634.6345;
    const frac2 = secondSeed - Math.floor(secondSeed);
    return {
      width: 8 + frac * 4,
      depth: 8 + frac2 * 4,
    };
  };
  // Function to create 3D object mesh
  const createObjectMesh = (obj: CustomEnvironmentObject): THREE.Group => {
    const group = new THREE.Group();
    const { type, height, x, z } = obj;
    
    if (type === 'building') {
      // Main structure - match old building style
      const industryPalette = [0x3a5f78, 0x436f8b, 0x447a6e, 0x746b4d, 0x5e4f74, 0x7e4b56];
      const baseColor = industryPalette[Math.floor(Math.random() * industryPalette.length)];
      const footprint = getStableBuildingFootprint(x, z, height);
      const width = obj.width ?? footprint.width;
      const depth = obj.depth ?? footprint.depth;
      
      const struct = new THREE.Mesh(
        new THREE.BoxGeometry(width, height, depth),
        new THREE.MeshStandardMaterial({
          color: baseColor,
          roughness: 0.52,
          metalness: 0.48,
          emissive: 0x0b2032,
          emissiveIntensity: 0.12,
        }),
      );
      struct.position.set(0, height / 2, 0);
      struct.castShadow = true;
      struct.receiveShadow = true;
      group.add(struct);
      
      // Roof ring
      const roofRing = new THREE.Mesh(
        new THREE.RingGeometry(Math.min(width, depth) * 0.16, Math.min(width, depth) * 0.35, 20),
        new THREE.MeshBasicMaterial({ color: 0x19d4ff, transparent: true, opacity: 0.18 }),
      );
      roofRing.rotation.x = -Math.PI / 2;
      roofRing.position.set(0, height + 0.65, 0);
      group.add(roofRing);
      
      // Side tank
      const sideTank = new THREE.Mesh(
        new THREE.CylinderGeometry(1.8, 1.8, Math.max(6, height * 0.45), 14),
        new THREE.MeshStandardMaterial({ color: 0x8ea3b4, roughness: 0.38, metalness: 0.58, emissive: 0x1e2328, emissiveIntensity: 0.1 }),
      );
      sideTank.position.set(width * 0.34, Math.max(6, height * 0.45) / 2, -depth * 0.34);
      sideTank.castShadow = true;
      group.add(sideTank);
      
      // Catwalk
      const catwalk = new THREE.Mesh(
        new THREE.BoxGeometry(width * 0.78, 0.35, 1.4),
        new THREE.MeshStandardMaterial({ color: 0x94b7c9, roughness: 0.42, metalness: 0.55 }),
      );
      catwalk.position.set(0, height * 0.62, depth * 0.52);
      group.add(catwalk);
      
      // Warning lamp
      const warningLamp = new THREE.PointLight(Math.random() > 0.5 ? 0xff8a3d : 0x24d2ff, 0.7, 60);
      warningLamp.position.set(0, height + 5, 0);
      group.add(warningLamp);
      
      
    } else if (type === 'machine') {
      // Machine - cylinder with enhanced look
      const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(4, 4, height, 8),
        new THREE.MeshStandardMaterial({color: 0x00ffff, emissive: 0x00ffff, emissiveIntensity: 0.3, roughness: 0.4, metalness: 0.6})
      );
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      group.add(mesh);
      
      // Glow light
      const glow = new THREE.PointLight(0x00ffff, 0.5, 50);
      glow.position.set(0, height / 2, 0);
      group.add(glow);
      
    } else if (type === 'pipe') {
      // Pipe - thin cylinder with glow
      const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(1.5, 1.5, height, 6),
        new THREE.MeshStandardMaterial({color: 0xffaa00, emissive: 0xffaa00, emissiveIntensity: 0.4, roughness: 0.3, metalness: 0.7})
      );
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      group.add(mesh);
      
      // Glow light
      const glow = new THREE.PointLight(0xff8a3d, 0.6, 40);
      glow.position.set(0, height / 2, 0);
      group.add(glow);
    }
    
    group.position.set(x, 0, z);
    return group;
  };

  // Function to update 3D environment
  const updateEnvironmentVisualization = () => {
    // Remove old custom object meshes without mutating the live array while iterating forward
    for (let i = customObjectsGroup.children.length - 1; i >= 0; i--) {
      const child = customObjectsGroup.children[i];
      if (child !== customObjectsGroup.children[0]) {
        customObjectsGroup.remove(child);
      }
    }
    
    // Clear custom building collision fields
    customBuildingFields = [];
    
    // Add all objects to scene and track buildings for collision
    environmentObjects.forEach(obj => {
      if (obj.type === 'building' && (obj.width === undefined || obj.depth === undefined)) {
        const footprint = getStableBuildingFootprint(obj.x, obj.z, obj.height);
        obj.width = footprint.width;
        obj.depth = footprint.depth;
      }

      const mesh = createObjectMesh(obj);
      obj.mesh = mesh;
      customObjectsGroup.add(mesh);
      
      // Track buildings for collision detection
      if (obj.type === 'building') {
        const width = obj.width ?? 8;
        const depth = obj.depth ?? 8;
        customBuildingFields.push({
          x: obj.x,
          z: obj.z,
          radius: Math.max(width, depth) * 0.72 + 6,
        });
      }
    });
  };

  const updateObjectList = () => {
    if (environmentObjects.length === 0) {
      objList.innerHTML = 'No objects added';
    } else {
      objList.innerHTML = '';
      environmentObjects.forEach((obj, idx) => {
        const colors: any = {building: '#00ffcc', machine: '#00ffff', pipe: '#ffaa00'};
        const itemDiv = document.createElement('div');
        itemDiv.style.cssText = `color: ${colors[obj.type]}; margin: 4px 0; display: flex; justify-content: space-between; align-items: center;`;
        
        const textSpan = document.createElement('span');
        textSpan.textContent = `${idx + 1}. ${obj.type.toUpperCase()} (h: ${obj.height}u) @ (${obj.x.toFixed(1)}, ${obj.z.toFixed(1)})`;
        itemDiv.appendChild(textSpan);
        
        const removeBtn = document.createElement('button');
        removeBtn.textContent = 'remove';
        removeBtn.style.cssText = 'background: none; border: none; color: #ff6666; cursor: pointer; font-size: 10px; padding: 0; margin: 0;';
        removeBtn.addEventListener('click', () => {
          environmentObjects.splice(idx, 1);
          updateObjectList();
          updateEnvironmentVisualization();
        });
        itemDiv.appendChild(removeBtn);
        objList.appendChild(itemDiv);
      });
    }
  };

  // Button handlers
  buildingBtn.addEventListener('click', () => {
    selectedObjectType = 'building';
    buildingBtn.style.background = '#003a4a';
    machineBtn.style.background = '#1a3a4a';
    pipeBtn.style.background = '#1a3a4a';
    coordDiv.innerHTML = '<strong>Mouse Coords:</strong> X: 0.0, Z: 0.0 | <strong>Mode:</strong> <span style="color: #00ffcc;">Building</span>';
  });

  machineBtn.addEventListener('click', () => {
    selectedObjectType = 'machine';
    buildingBtn.style.background = '#1a3a4a';
    machineBtn.style.background = '#003a4a';
    pipeBtn.style.background = '#1a3a4a';
    coordDiv.innerHTML = '<strong>Mouse Coords:</strong> X: 0.0, Z: 0.0 | <strong>Mode:</strong> <span style="color: #00ffff;">Machine</span>';
  });

  pipeBtn.addEventListener('click', () => {
    selectedObjectType = 'pipe';
    buildingBtn.style.background = '#1a3a4a';
    machineBtn.style.background = '#1a3a4a';
    pipeBtn.style.background = '#3a2a1a';
    coordDiv.innerHTML = '<strong>Mouse Coords:</strong> X: 0.0, Z: 0.0 | <strong>Mode:</strong> <span style="color: #ffaa00;">Pipe</span>';
  });

  heightInput.addEventListener('input', () => {
    const h = parseInt(heightInput.value);
    heightVal.textContent = `${h} units`;
  });

  // Create invisible ground plane for raycasting
  const groundPlane = new THREE.Mesh(
    new THREE.PlaneGeometry(WORLD_SIZE * 2.5, WORLD_SIZE * 2.5),
    new THREE.MeshBasicMaterial({transparent: true, opacity: 0})
  );
  groundPlane.rotation.x = -Math.PI / 2;
  groundPlane.position.y = 0.01;
  customObjectsGroup.add(groundPlane);

  // Click to place on canvas
  canvas.addEventListener('click', (e: MouseEvent) => {
    if (!selectedObjectType) return;
    
    // Calculate mouse position
    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

    // Raycasting
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(groundPlane);

    if (intersects.length > 0) {
      const point = intersects[0].point;
      const x = point.x;
      const z = point.z;
      const height = parseInt(heightInput.value);

      environmentObjects.push({
        type: selectedObjectType,
        height: height,
        x: x,
        z: z
      });
      updateObjectList();
      updateEnvironmentVisualization();
      console.log(`Added ${selectedObjectType} at (${x.toFixed(1)}, ${z.toFixed(1)}) height ${height}`);
    }
  });

  // Track mouse movement for coordinate display
  canvas.addEventListener('mousemove', (e: MouseEvent) => {
    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(groundPlane);

    if (intersects.length > 0) {
      const point = intersects[0].point;
      const modeColors: any = {building: '#00ffcc', machine: '#00ffff', pipe: '#ffaa00'};
      const modeColor = modeColors[selectedObjectType];
      coordDiv.innerHTML = `<strong>Mouse Coords:</strong> X: ${point.x.toFixed(1)}, Z: ${point.z.toFixed(1)} | <strong>Mode:</strong> <span style="color: ${modeColor};">${selectedObjectType.toUpperCase()}</span>`;
    }
  });

  clearBtn.addEventListener('click', () => {
    if (confirm('Clear all custom objects?')) {
      environmentObjects = [];
      updateObjectList();
      updateEnvironmentVisualization();
    }
  });

  clearBuildingsBtn.addEventListener('click', () => {
    const buildingCount = environmentObjects.filter(obj => obj.type === 'building').length;
    if (buildingCount === 0) {
      alert('No buildings found in the current environment.');
      return;
    }
    if (confirm(`Remove ${buildingCount} building object(s) and keep the ground / other objects?`)) {
      environmentObjects = environmentObjects.filter(obj => obj.type !== 'building');
      updateObjectList();
      updateEnvironmentVisualization();
    }
  });

  saveBtn.addEventListener('click', () => {
    localStorage.setItem('default_environment', JSON.stringify(environmentObjects));
    alert(`Saved ${environmentObjects.length} objects. Ready to test drone!`);
  });

  loadBtn.addEventListener('click', () => {
    const saved = localStorage.getItem('default_environment');
    if (saved) {
      try {
        environmentObjects = JSON.parse(saved);
        updateObjectList();
        updateEnvironmentVisualization();
        alert(`Loaded ${environmentObjects.length} objects`);
      } catch (e) {
        alert('Error loading environment');
      }
    } else {
      alert('No saved environment found');
    }
  });

  applyPresetBtn.addEventListener('click', () => {
    const preset = presetSelect.value;
    if (preset === 'current') {
      environmentObjects = serializeEnvironmentObjects(loadSavedEnvironmentObjects());
    } else {
      environmentObjects = serializeEnvironmentObjects(buildPresetEnvironmentObjects(preset));
    }
    updateObjectList();
    updateEnvironmentVisualization();
    localStorage.setItem('default_environment', JSON.stringify(environmentObjects));
    trainingStatus.textContent = `Applied preset: ${preset}\nObjects in environment: ${environmentObjects.length}`;
  });

  const environmentSpecFromCurrentState = (preset: string) => {
    const activeObjects = environmentObjects.length > 0 ? environmentObjects : loadSavedEnvironmentObjects();
    return {
      project_id: (projectInput.value || 'project01').trim(),
      source: environmentObjects.length > 0 ? 'live-builder' : 'saved-localstorage',
      preset,
      grid: {
        width: 40,
        height: 40,
        cell_size: 3,
      },
      rendering: {
        wall_height: Math.max(8, Math.round(activeObjects.reduce((maxHeight, obj) => Math.max(maxHeight, obj.height), 12))),
        floor_material: 'concrete',
        wall_material: 'steel',
        lighting_profile: 'industrial-default',
        fog_density: 0.004,
      },
      optimization: {
        merge_static_geometry: true,
        use_instancing: true,
        enable_lod: true,
        target_fps: 60,
        max_draw_calls: Math.max(120, 260 - activeObjects.length * 4),
      },
      objects: serializeEnvironmentObjects(activeObjects),
    };
  };

  trainingLaunchBtn.addEventListener('click', async () => {
    const projectId = (projectInput.value || 'project01').trim();
    const mapImage = (mapInput.value || 'sample_map.png').trim();
    const useDefaultKey = useDefaultSelect.value === 'true';
    const token = tokenInput.value.trim();
    const trainingMode = trainingModeSelect.value;
    const environmentName = presetSelect.value;
    const environmentSpec = environmentSpecFromCurrentState(environmentName);
    const trainSeconds = parseInt(trainSecondsInput.value || '5400');

    const payload = {
      mapImage,
      projectId,
      useDefaultKey,
      token,
      trainSeconds: String(trainSeconds),
      planningSteps: planningStepsInput.value || '60',
      episodes: episodesInput.value || '15000',
      waitKernel: true,
      environmentName,
      environmentSpec,
    };

    const endpoint = trainingMode === 'local' ? `${API_BASE_URL}/api/local/start` : `${API_BASE_URL}/api/kaggle/start`;
    const summaryMode = trainingMode === 'local' ? 'LOCAL' : 'KAGGLE';

    trainingStatus.textContent = `Starting ${summaryMode} training...\nProject: ${projectId}\nPreset: ${environmentName}\nObjects: ${environmentSpec.objects.length}`;

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const startData = await response.json();
      const runId = startData.id;
      
      if (!runId) {
        trainingStatus.textContent = `Error: No run ID returned\n${JSON.stringify(startData)}`;
        return;
      }

      // Start polling for log updates
      let isTrainingActive = true;
      let pollCount = 0;
      const pollInterval = setInterval(async () => {
        try {
          const logResponse = await fetch(`${API_BASE_URL}/api/kaggle/run/${runId}/log`);
          const logData = await logResponse.json();
          const { log, meta } = logData;
          pollCount++;

          if (!meta) {
            trainingStatus.textContent = `Error: No metadata received`;
            clearInterval(pollInterval);
            return;
          }

          // Calculate elapsed time and progress
          const createdAt = new Date(meta.createdAt).getTime();
          const now = Date.now();
          const elapsedSeconds = Math.floor((now - createdAt) / 1000);
          const elapsedMinutes = Math.floor(elapsedSeconds / 60);
          const elapsedHours = Math.floor(elapsedMinutes / 60);
          
          const progressPercent = Math.min(100, Math.round((elapsedSeconds / trainSeconds) * 100));
          const timeRemain = Math.max(0, trainSeconds - elapsedSeconds);
          const remainMinutes = Math.floor(timeRemain / 60);
          const remainSeconds = timeRemain % 60;

          // Extract last N lines from log for display
          const logLines = log.split('\n');
          const lastLines = logLines.slice(-12).join('\n');

          let statusText = `[${summaryMode} TRAINING IN PROGRESS]\n`;
          statusText += `Run ID: ${runId}\n`;
          statusText += `Status: ${meta.status || 'unknown'}\n`;
          statusText += `Elapsed: ${String(elapsedHours).padStart(2,'0')}:${String(elapsedMinutes % 60).padStart(2,'0')}:${String(elapsedSeconds % 60).padStart(2,'0')} / ${Math.floor(trainSeconds / 3600)}h${Math.floor((trainSeconds % 3600) / 60)}m\n`;
          statusText += `Progress: ${progressPercent}% [${Array(Math.floor(progressPercent / 5)).fill('=').join('')}${Array(20 - Math.floor(progressPercent / 5)).fill('-').join('')}]\n`;
          if (timeRemain > 0) {
            statusText += `ETA: ${String(remainMinutes).padStart(2,'0')}:${String(remainSeconds).padStart(2,'0')} remaining\n`;
          }
          statusText += `\n--- Recent Logs (Last 12 lines) ---\n${lastLines}`;

          trainingStatus.textContent = statusText;

          // Check if training is complete
          if (meta.status === 'complete' || meta.status === 'error') {
            clearInterval(pollInterval);
            isTrainingActive = false;

            // Display completion details
            let completionText = `[${summaryMode.toUpperCase()} TRAINING COMPLETE]\n`;
            completionText += `Run ID: ${runId}\n`;
            completionText += `Status: ${meta.status.toUpperCase()}\n`;
            completionText += `Exit Code: ${meta.exitCode !== undefined ? meta.exitCode : 'N/A'}\n`;
            
            const finishedAt = meta.finishedAt ? new Date(meta.finishedAt).getTime() : now;
            const totalElapsed = Math.floor((finishedAt - createdAt) / 1000);
            const totalHours = Math.floor(totalElapsed / 3600);
            const totalMins = Math.floor((totalElapsed % 3600) / 60);
            const totalSecs = totalElapsed % 60;
            completionText += `Total Time: ${String(totalHours).padStart(2,'0')}:${String(totalMins).padStart(2,'0')}:${String(totalSecs).padStart(2,'0')}\n`;
            
            if (meta.policyOutputPath) {
              completionText += `Policy Saved: ${meta.policyOutputPath.split(/[\\\/]/).pop()}\n`;
            }
            
            completionText += `\n--- Full Training Logs ---\n${log}`;
            
            if (meta.status === 'complete' && meta.policyOutputPath) {
              completionText += `\n\n[ACTIONS]\nPolicy generated and ready to activate.\n`;
            }
            
            trainingStatus.textContent = completionText;
          }
        } catch (pollErr) {
          trainingStatus.textContent = `Polling error: ${String(pollErr)}\nWill continue polling...`;
        }
      }, 1500); // Poll every 1.5 seconds

    } catch (err) {
      trainingStatus.textContent = `Failed to start training:\n${String(err)}`;
    }
  });
}

const clock = new THREE.Clock();
void loadHybridPolicy();
let lastVisualRefresh = 0;

function animate(): void {
  const dt = Math.min(clock.getDelta(), 0.035);
  const elapsed = clock.getElapsedTime();

  if (missionActive && !missionFinalized) {
    missionTimeRemaining = Math.max(0, missionTimeRemaining - dt);
    if (missionTimeRemaining <= 0) {
      finalizeMission('countdown expired');
    } else if (discoveredZones.length >= heatZoneCount && heatZoneCount > 0) {
      finalizeMission('all heat zones discovered');
    }
  }

  if (elapsed - lastVisualRefresh > 0.28) {
    lastVisualRefresh = elapsed;
    updateDiscoveredZonesVisuals();
  }
  visualizeThermalScan();
  updateDroneMovement(dt);

  if (elapsed - lastTelemetryUpdate > 0.2) {
    lastTelemetryUpdate = elapsed;

    // Update telemetry
    metrics.textContent = `AUTONOMOUS DRONE - EXPLORATION & DISCOVERY MISSION\nStatus: ${explorationMode ? `EXPLORING (${discoveredZones.length} zones found)` : droneState.missionComplete ? 'MISSION COMPLETE' : 'IN MISSION'}\nZones Discovered: ${discoveredZones.length}\nDistance: ${droneState.distanceTraveled.toFixed(1)} m`;

    drlMetrics.textContent = `Algorithm: ${policyStatus === 'ready' ? 'Hybrid Dyna Policy Inference' : 'RL Environment Fallback Controller'}\nCoverage Strategy: RL Policy + Thermal Guidance\nPolicy Mix: 100% RL\nPhase: TRAINING EPISODE\nMission Countdown: ${Math.ceil(missionTimeRemaining)} sec\nScan Radius: ${SCAN_RADIUS} units | Speed: ${droneState.maxSpeed} m/s\nPolicy Status: ${policyStatus.toUpperCase()} | Policy Grid: ${policyGridX}x${policyGridZ}\nThermal Readings: ${thermalReadings.length}\nCumulative Reward: ${cumulativeReward.toFixed(3)} | Episode Steps: ${episodeSteps}\nThermal Gradient: ${(currentThermalSignal - lastThermalSignal).toFixed(4)} | Epsilon: ${epsilon.toFixed(4)}\nVelocity: ${droneState.velocity.length().toFixed(2)} m/s${lastCollision ? '\nCollision detected: auto-reset triggered' : ''}`;

    // Update UI values
    if (droneStatusEl)
      droneStatusEl.textContent = explorationMode
        ? 'EXPLORING'
        : droneState.isFlying
          ? 'IN MISSION'
          : droneState.missionComplete
            ? 'COMPLETE'
            : 'READY';
    if (zonesDiscoveredEl) zonesDiscoveredEl.textContent = String(discoveredZones.length);
    if (zonesVisitedEl) zonesVisitedEl.textContent = 'Coordinates only';
    if (missionCountdownEl) {
      const minutes = Math.floor(missionTimeRemaining / 60);
      const seconds = Math.floor(missionTimeRemaining % 60).toString().padStart(2, '0');
      missionCountdownEl.textContent = `${minutes.toString().padStart(2, '0')}:${seconds}`;
    }
    if (distanceValEl) distanceValEl.textContent = `${droneState.distanceTraveled.toFixed(1)} m`;
    if (speedValEl) speedValEl.textContent = `${droneState.maxSpeed.toFixed(1)} m/s`;
  }

  // Update camera based on selected view
  updateCameraView();

  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

setupAppUI();
setupUI();
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
