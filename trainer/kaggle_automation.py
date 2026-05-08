"""
Kaggle GPU Training Automation API Service

Exposes Flask endpoints for:
  - POST /api/run_full_automation: Trigger Kaggle GPU training
  - GET /api/automation-status: Check automation job status
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path
import json
import traceback
import logging
import uuid
import os
import threading
import subprocess
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable cross-origin requests for frontend
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Job management for automation runs
REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = REPO_ROOT / 'shared' / 'automation_jobs'
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _write_job_meta(job_id: str, meta: dict):
    """Write job metadata to JSON file."""
    path = JOBS_DIR / f"{job_id}.json"
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2, default=str)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Kaggle Automation API",
        "version": "0.1.0"
    }), 200


@app.route('/api/run_full_automation', methods=['POST'])
def run_full_automation():
    """
    Trigger the full Kaggle GPU training automation pipeline.

    Accepts multipart form-data with:
      - project_id (required)
      - client_name (required)
      - kaggle_token (required)
      - critical_temperature (optional)

    Returns a job_id which can be polled via /api/automation-status
    """
    try:
        form = request.form
        project_id = form.get('project_id')
        client_name = form.get('client_name')
        kaggle_token = form.get('kaggle_token') or form.get('token')

        if not project_id or not client_name or not kaggle_token:
            return jsonify({"error": "Missing required fields: project_id, client_name, kaggle_token"}), 400

        # Build command to run full_automation.py
        script_path = REPO_ROOT / 'pipeline' / 'full_automation.py'
        if not script_path.exists():
            return jsonify({"error": "Automation script not found"}), 500

        job_id = uuid.uuid4().hex
        stdout_path = JOBS_DIR / f"{job_id}.out"
        stderr_path = JOBS_DIR / f"{job_id}.err"

        cmd = [sys.executable, str(script_path),
               '--repo-root', str(REPO_ROOT),
               '--project-id', project_id,
               '--client-name', client_name,
               '--wait-kernel']

        env = os.environ.copy()
        env['KAGGLE_API_TOKEN'] = kaggle_token

        # Start process
        out_f = open(stdout_path, 'w', encoding='utf-8')
        err_f = open(stderr_path, 'w', encoding='utf-8')
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, env=env, cwd=str(REPO_ROOT))

        # Write initial job meta
        meta = {
            'job_id': job_id,
            'pid': proc.pid,
            'status': 'running',
            'cmd': cmd,
            'started_at': datetime.now().isoformat(),
            'stdout': str(stdout_path),
            'stderr': str(stderr_path)
        }
        _write_job_meta(job_id, meta)

        # Background thread to wait and update status
        def _wait_and_update(p: subprocess.Popen, job_id: str):
            rc = p.wait()
            meta_path = JOBS_DIR / f"{job_id}.json"
            try:
                with open(meta_path, 'r', encoding='utf-8') as fh:
                    current = json.load(fh)
            except Exception:
                current = {}
            current.update({
                'status': 'finished' if rc == 0 else 'failed',
                'returncode': rc,
                'finished_at': datetime.now().isoformat()
            })
            with open(meta_path, 'w', encoding='utf-8') as fh:
                json.dump(current, fh, indent=2, default=str)
            try:
                out_f.close()
                err_f.close()
            except Exception:
                pass

        thread = threading.Thread(target=_wait_and_update, args=(proc, job_id), daemon=True)
        thread.start()

        return jsonify({'job_id': job_id, 'status': 'started'}), 202

    except Exception as e:
        logger.error(f"Error starting automation: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/automation-status', methods=['GET'])
def automation_status():
    """Get status and logs for an automation job."""
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    meta_path = JOBS_DIR / f"{job_id}.json"
    if not meta_path.exists():
        return jsonify({"error": "job not found"}), 404
    
    try:
        with open(meta_path, 'r', encoding='utf-8') as fh:
            meta = json.load(fh)
        
        # Tail stdout/stderr last lines
        def _tail(path, lines=20):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = f.read().splitlines()
                    return "\n".join(data[-lines:])
            except Exception:
                return ""

        meta['stdout_tail'] = _tail(meta.get('stdout', ''), 40)
        meta['stderr_tail'] = _tail(meta.get('stderr', ''), 40)
        return jsonify(meta), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    """Run the Flask development server."""
    logger.info("Starting Kaggle Automation API on http://localhost:5001")
    logger.info("Automation endpoint: POST http://localhost:5001/api/run_full_automation")
    app.run(debug=True, port=5001, host='0.0.0.0')
