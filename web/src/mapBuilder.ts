/**
 * Map Builder from Occupancy Grid
 * 
 * Creates 3D scenes from occupancy matrices using Three.js.
 * 
 * Pipeline:
 *   1. Parse occupancy grid (2D matrix: 0=free, 1=wall)
 *   2. Create box geometries for wall cells
 *   3. Merge geometries for performance
 *   4. Apply materials and add to scene
 */

import * as THREE from 'three';
import { getBlueprintApiBaseUrl } from './runtimeConfig';

export interface OccupancyGrid {
  width: number;
  height: number;
  grid: number[][];  // 2D array
  flat?: number[];   // Optional flattened array
  metadata?: Record<string, any>;
}

export interface MapBuilderConfig {
  cellHeight: number;  // Height of vertical walls (units)
  cellWidth: number;   // XZ size of each cell (units)
  yOffset: number;     // Height offset from ground (units)
  wallMaterial?: THREE.Material;
  floorMaterial?: THREE.Material;
  showDebugGridlines?: boolean;
  gridlineColor?: number;
}

/**
 * Builds a 3D scene from an occupancy grid.
 */
export class MapBuilder {
  private config: MapBuilderConfig;
  private wallGroup: THREE.Group;
  private occupancyGrid: OccupancyGrid | null = null;
  
  constructor(config?: Partial<MapBuilderConfig>) {
    this.config = {
      cellHeight: 5,
      cellWidth: 1,
      yOffset: 0,
      wallMaterial: new THREE.MeshStandardMaterial({
        color: 0x666666,
        metalness: 0.3,
        roughness: 0.8,
      }),
      floorMaterial: new THREE.MeshStandardMaterial({
        color: 0xcccccc,
        metalness: 0.1,
        roughness: 0.9,
      }),
      showDebugGridlines: false,
      gridlineColor: 0x999999,
      ...config,
    };
    
    this.wallGroup = new THREE.Group();
    this.wallGroup.name = 'OccupancyMapWalls';
  }
  
  /**
   * Load and build occupancy grid into 3D scene.
   * 
   * @param grid - Occupancy grid data
   * @returns The wall group (ready to add to scene)
   */
  buildFromGrid(grid: OccupancyGrid): THREE.Group {
    this.occupancyGrid = grid;
    
    // Clear existing geometry
    this.wallGroup.clear();
    
    const cellW = this.config.cellWidth;
    const cellH = this.config.cellHeight;
    const offset = this.config.yOffset;
    
    // Create geometries for all wall cells
    const wallGeometries: THREE.BoxGeometry[] = [];
    
    const gridData = grid.grid || [];
    const height = gridData.length;
    const width = gridData[0]?.length || 0;
    
    if (height === 0 || width === 0) {
      console.warn('Invalid occupancy grid dimensions');
      return this.wallGroup;
    }
    
    // Iterate through grid and collect wall positions
    for (let z = 0; z < height; z++) {
      for (let x = 0; x < width; x++) {
        const cell = gridData[z]?.[x] || 0;
        
        // 1 = wall/obstacle, 0 = free space
        if (cell === 1) {
          const boxGeom = new THREE.BoxGeometry(cellW, cellH, cellW);
          
          // Position: center of cell
          const posX = (x - width / 2) * cellW;
          const posZ = (z - height / 2) * cellW;
          const posY = offset + cellH / 2;
          
          // Create matrix for positioning
          const matrix = new THREE.Matrix4();
          matrix.setPosition(posX, posY, posZ);
          
          // Apply matrix to geometry to avoid creating meshes
          boxGeom.applyMatrix4(matrix);
          
          wallGeometries.push(boxGeom);
        }
      }
    }
    
    // Merge all geometries into a single buffer geometry
    if (wallGeometries.length > 0) {
      const mergedGeometry = this.mergeGeometries(wallGeometries);
      
      const wallMesh = new THREE.Mesh(mergedGeometry, this.config.wallMaterial);
      wallMesh.castShadow = true;
      wallMesh.receiveShadow = true;
      wallMesh.name = 'OccupancyMapMesh';
      
      this.wallGroup.add(wallMesh);
      
      console.log(
        `Built map from occupancy grid: ${width}x${height} → ${wallGeometries.length} wall cells`
      );
    }
    
    // Optional: Add debug gridlines
    if (this.config.showDebugGridlines) {
      this.addDebugGridlines(width, height);
    }
    
    // Metadata logging
    const metadata = grid.metadata || {};
    console.log(
      `Occupancy: ${metadata.obstacle_ratio || '?'}% walls, ${metadata.free_ratio || '?'}% free`
    );
    
    return this.wallGroup;
  }
  
  /**
   * Merge multiple geometries into single buffer geometry.
   * Fallback if Three.js BufferGeometryUtils not available.
   */
  private mergeGeometries(geometries: THREE.BoxGeometry[]): THREE.BufferGeometry {
    const merged = new THREE.BufferGeometry();
    
    let vertexCount = 0;
    const vertices: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];
    
    for (const geom of geometries) {
      // Get vertex and index data
      const v = geom.getAttribute('position')?.array || new Float32Array();
      const n = geom.getAttribute('normal')?.array || new Float32Array();
      const idx = geom.getIndex()?.array || new Uint32Array();
      
      // Append vertices
      for (let i = 0; i < v.length; i++) {
        vertices.push((v as any)[i]);
        normals.push((n as any)[i]);
      }
      
      // Append indices with offset
      for (let i = 0; i < idx.length; i++) {
        indices.push((idx as any)[i] + vertexCount);
      }
      
      vertexCount += (v.length / 3);
    }
    
    merged.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices), 3));
    merged.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(normals), 3));
    merged.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
    
    return merged;
  }
  
  /**
   * Add debug gridlines for visualization.
   */
  private addDebugGridlines(width: number, height: number) {
    const cellW = this.config.cellWidth;
    const material = new THREE.LineBasicMaterial({ color: this.config.gridlineColor });
    
    const halfW = (width / 2) * cellW;
    const halfH = (height / 2) * cellW;
    const y = this.config.yOffset + this.config.cellHeight;
    
    // Create vertices for grid
    const vertices: THREE.Vector3[] = [];
    
    // Draw grid pattern
    for (let i = 0; i <= width; i++) {
      const x = -halfW + i * cellW;
      vertices.push(new THREE.Vector3(x, y, -halfH));
      vertices.push(new THREE.Vector3(x, y, halfH));
    }
    
    for (let j = 0; j <= height; j++) {
      const z = -halfH + j * cellW;
      vertices.push(new THREE.Vector3(-halfW, y, z));
      vertices.push(new THREE.Vector3(halfW, y, z));
    }
    
    // Create buffer geometry from vertices
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(vertices.length * 3);
    for (let i = 0; i < vertices.length; i++) {
      positions[i * 3] = vertices[i].x;
      positions[i * 3 + 1] = vertices[i].y;
      positions[i * 3 + 2] = vertices[i].z;
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    const lines = new THREE.LineSegments(geometry, material);
    this.wallGroup.add(lines);
    console.log('Added debug gridlines');
  }
  
  /**
   * Get bounding box of the generated map.
   */
  getBoundingBox(): THREE.Box3 {
    return new THREE.Box3().setFromObject(this.wallGroup);
  }
  
  /**
   * Get the wall group (add to scene).
   */
  getWallGroup(): THREE.Group {
    return this.wallGroup;
  }
  
  /**
   * Clear all meshes and geometry.
   */
  clear() {
    this.wallGroup.clear();
    this.occupancyGrid = null;
  }
  
  /**
   * Export current grid as JSON.
   */
  exportGrid(): string {
    if (!this.occupancyGrid) {
      return '{}';
    }
    return JSON.stringify(this.occupancyGrid, null, 2);
  }
}

/**
 * Helper: Load occupancy grid from API and build map.
 */
export async function loadBlueprintAndBuild(
  file: File,
  mapBuilder: MapBuilder,
  apiUrl: string = getBlueprintApiBaseUrl()
): Promise<OccupancyGrid | null> {
  try {
    const formData = new FormData();
    formData.append('blueprint', file);
    
    const response = await fetch(`${apiUrl}/api/process-blueprint`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const error = await response.json();
      console.error('API error:', error);
      return null;
    }
    
    const result = await response.json();
    const grid = result.occupancy_grid;
    
    // Build 3D map
    mapBuilder.buildFromGrid(grid);
    
    console.log('Blueprint processed and map built successfully');
    return grid;
  } catch (error) {
    console.error('Failed to load and build blueprint:', error);
    return null;
  }
}

/**
 * Helper: Load test grid from API.
 */
export async function loadTestGrid(
  mapBuilder: MapBuilder,
  apiUrl: string = getBlueprintApiBaseUrl()
): Promise<OccupancyGrid | null> {
  try {
    const response = await fetch(`${apiUrl}/api/test-grid`);
    
    if (!response.ok) {
      console.error('Failed to load test grid');
      return null;
    }
    
    const result = await response.json();
    const grid = result.test_grid;
    
    mapBuilder.buildFromGrid(grid);
    console.log('Test grid loaded and map built');
    
    return grid;
  } catch (error) {
    console.error('Failed to load test grid:', error);
    return null;
  }
}
