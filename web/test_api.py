#!/usr/bin/env python3
"""
API Test Script - Verify Blueprint Processing API endpoints
"""

import os
import requests
import json
import time

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5001")

def test_health():
    """Test health endpoint."""
    print("\n[1] Testing /health endpoint...")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"✓ Status: {data.get('status')}")
            print(f"✓ Service: {data.get('service')}")
            return True
        else:
            print(f"✗ Status code: {r.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_config():
    """Test blueprint config endpoint."""
    print("\n[2] Testing /api/blueprint-config endpoint...")
    try:
        r = requests.get(f"{BASE_URL}/api/blueprint-config", timeout=5)
        if r.status_code == 200:
            data = r.json()
            config = data.get('processor_config', {})
            print(f"✓ Target resolution: {config.get('target_resolution')}")
            print(f"✓ Threshold block size: {config.get('threshold_block_size')}")
            print(f"✓ Supported formats: {data.get('supported_formats')}")
            return True
        else:
            print(f"✗ Status code: {r.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_grid():
    """Test test-grid endpoint."""
    print("\n[3] Testing /api/test-grid endpoint...")
    try:
        r = requests.get(f"{BASE_URL}/api/test-grid", timeout=5)
        if r.status_code == 200:
            data = r.json()
            if data.get('success'):
                grid = data.get('test_grid', {})
                print(f"✓ Test grid returned successfully")
                print(f"✓ Grid size: {len(grid.get('grid', []))}x{len(grid.get('grid', [[]])[0]) if grid.get('grid') else 0}")
                print(f"✓ Has metadata: {'metadata' in grid}")
                return True
            else:
                print(f"✗ Response not successful: {data}")
                return False
        else:
            print(f"✗ Status code: {r.status_code}")
            print(f"Response: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_process_blueprint():
    """Test blueprint processing with a test image."""
    print("\n[4] Testing /api/process-blueprint endpoint...")
    try:
        # Create a simple test image
        import numpy as np
        from PIL import Image
        import io
        
        # Create white image with black rectangle (wall)
        img_array = np.ones((200, 200, 3), dtype=np.uint8) * 255
        img_array[50:150, 50:150] = 0  # Black square
        
        # Convert to PIL and save to bytes
        img = Image.fromarray(img_array, 'RGB')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Upload
        files = {'blueprint': ('test.png', img_bytes, 'image/png')}
        r = requests.post(f"{BASE_URL}/api/process-blueprint", files=files, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            if data.get('success'):
                grid = data.get('occupancy_grid', {})
                print(f"✓ Blueprint processed successfully")
                print(f"✓ Filename: {data.get('filename')}")
                print(f"✓ Grid shape: {grid.get('width')}x{grid.get('height')}")
                print(f"✓ Has metadata: {'metadata' in grid}")
                return True
            else:
                print(f"✗ Processing failed: {data}")
                return False
        else:
            print(f"✗ Status code: {r.status_code}")
            print(f"Response: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*70)
    print("DRL WILDFIRE - API TEST SUITE")
    print("="*70)
    print(f"Base URL: {BASE_URL}")
    
    # Wait for API to be ready
    print("\nWaiting for API to be ready...")
    for i in range(10):
        try:
            requests.get(f"{BASE_URL}/health", timeout=2)
            print("✓ API is ready!")
            break
        except:
            if i < 9:
                print(f"  Attempt {i+1}/10... retrying in 1s")
                time.sleep(1)
            else:
                print("✗ API not responding. Make sure it's running on port 5001")
                return False
    
    # Run tests
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Config", test_config()))
    results.append(("Test Grid", test_grid()))
    results.append(("Process Blueprint", test_process_blueprint()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! API is working correctly.")
        return True
    else:
        print(f"\n✗ {total - passed} test(s) failed. Check output above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
