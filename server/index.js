const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

const app = express();
app.use(cors());
app.use(express.json());

const repoRoot = path.resolve(__dirname, '..');
const runsDir = path.join(__dirname, 'runs');
if (!fs.existsSync(runsDir)) fs.mkdirSync(runsDir, { recursive: true });

function writeRunMeta(runId, meta) {
  fs.writeFileSync(path.join(runsDir, `${runId}.json`), JSON.stringify(meta, null, 2));
}

app.get('/api/policy/current', (req, res) => {
  const policyPath = path.join(repoRoot, 'trainer', 'models', 'dyna_q_policy.json');
  if (!fs.existsSync(policyPath)) return res.json({ exists: false });
  const st = fs.statSync(policyPath);
  return res.json({ exists: true, path: policyPath, mtime: st.mtimeMs, size: st.size });
});

app.get('/api/policies', (req, res) => {
  const out = [];
  const trainerModels = path.join(repoRoot, 'trainer', 'models');
  if (fs.existsSync(trainerModels)) {
    fs.readdirSync(trainerModels).forEach(f => {
      if (f.endsWith('.json')) out.push(path.join(trainerModels, f));
    });
  }
  const localRuns = path.join(repoRoot, 'trainer', 'models', 'local_runs');
  if (fs.existsSync(localRuns)) {
    fs.readdirSync(localRuns).forEach(f => {
      if (f.endsWith('.json')) out.push(path.join(localRuns, f));
    });
  }
  const shared = path.join(repoRoot, 'shared', 'projects');
  if (fs.existsSync(shared)) {
    fs.readdirSync(shared).forEach(p => {
      const t = path.join(shared, p, 'training', 'dyna_q_policy.json');
      if (fs.existsSync(t)) out.push(t);
    });
  }
  res.json(out);
});

app.get('/api/policies/content', (req, res) => {
  const rawPath = String(req.query.path || '').trim();
  if (!rawPath) return res.status(400).json({ error: 'path required' });

  const allowedRoots = [
    path.join(repoRoot, 'trainer', 'models'),
    path.join(repoRoot, 'web', 'public', 'models'),
  ].map((dir) => path.resolve(dir));

  let resolved = rawPath;
  if (!path.isAbsolute(resolved)) {
    resolved = path.join(repoRoot, resolved);
  }
  resolved = path.resolve(resolved);

  const isAllowed = allowedRoots.some((root) => resolved === root || resolved.startsWith(root + path.sep));
  if (!isAllowed) {
    return res.status(403).json({ error: 'policy path not allowed' });
  }
  if (!fs.existsSync(resolved)) {
    return res.status(404).json({ error: 'policy not found' });
  }

  try {
    const payload = JSON.parse(fs.readFileSync(resolved, 'utf8'));
    return res.json({ path: resolved, payload });
  } catch (err) {
    return res.status(500).json({ error: `failed to read policy: ${String(err)}` });
  }
});

app.post('/api/policies/activate', (req, res) => {
  const { path: src } = req.body;
  if (!src || !fs.existsSync(src)) return res.status(400).json({ error: 'source not found' });
  const dest = path.join(repoRoot, 'trainer', 'models', 'dyna_q_policy.json');
  const backups = path.join(repoRoot, 'trainer', 'models', 'backups');
  if (!fs.existsSync(backups)) fs.mkdirSync(backups, { recursive: true });
  if (fs.existsSync(dest)) {
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    fs.copyFileSync(dest, path.join(backups, `dyna_q_policy.pre_switch.${stamp}.json`));
  }
  fs.copyFileSync(src, dest);
  return res.json({ ok: true, activated: dest });
});

app.post('/api/kaggle/start', (req, res) => {
  const body = req.body || {};
  const mapImage = body.mapImage;
  const projectId = body.projectId;
  const useDefaultKey = !!body.useDefaultKey;
  const token = body.token || '';
  const trainSeconds = body.trainSeconds || '5400';
  const planningSteps = body.planningSteps || '60';
  const episodes = body.episodes || '15000';
  const waitKernel = !!body.waitKernel;
  const environmentName = body.environmentName || 'current';
  const environmentSpec = body.environmentSpec || null;

  if (!mapImage || !projectId) return res.status(400).json({ error: 'mapImage and projectId required' });

  const runId = uuidv4();
  const logPath = path.join(runsDir, `${runId}.log`);
  const meta = { id: runId, status: 'starting', createdAt: Date.now(), mapImage, projectId, pid: null, logPath, environmentName };
  writeRunMeta(runId, meta);

  if (environmentSpec) {
    fs.writeFileSync(path.join(runsDir, `${runId}.environment.json`), JSON.stringify(environmentSpec, null, 2));
  }

  // Build args
  const args = ['pipeline/start_kaggle_training.py', '--map-image', mapImage, '--project-id', projectId, '--repo-root', repoRoot];
  if (useDefaultKey) args.push('--use-default-key');
  if (token) { args.push('--kaggle-token', token); }
  if (waitKernel) args.push('--wait-kernel');

  const py = process.env.PYTHON || 'python';
  const env = Object.assign({}, process.env, {
    DYNA_TRAIN_SECONDS: String(trainSeconds),
    DYNA_PLANNING_STEPS: String(planningSteps),
    DYNA_EPISODES: String(episodes),
    TRAINING_ENVIRONMENT_NAME: environmentName,
    TRAINING_ENVIRONMENT_SPEC_PATH: environmentSpec ? path.join(runsDir, `${runId}.environment.json`) : '',
  });

  const outStream = fs.createWriteStream(logPath, { flags: 'a' });

  const child = spawn(py, args, { cwd: repoRoot, env });
  meta.pid = child.pid;
  meta.status = 'running';
  writeRunMeta(runId, meta);

  child.stdout.on('data', (d) => { outStream.write(d); });
  child.stderr.on('data', (d) => { outStream.write(d); });

  child.on('close', (code) => {
    meta.status = code === 0 ? 'complete' : 'error';
    meta.exitCode = code;
    meta.finishedAt = Date.now();
    writeRunMeta(runId, meta);
    outStream.end();
  });

  res.json({ id: runId, logPath });
});

app.post('/api/local/start', (req, res) => {
  const body = req.body || {};
  const projectId = body.projectId || 'local-project';
  const trainSeconds = body.trainSeconds || '5400';
  const planningSteps = body.planningSteps || '60';
  const episodes = body.episodes || '15000';
  const environmentName = body.environmentName || 'current';
  const environmentSpec = body.environmentSpec || null;

  const runId = uuidv4();
  const logPath = path.join(runsDir, `${runId}.log`);
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const localPolicyDir = path.join(repoRoot, 'trainer', 'models', 'local_runs');
  if (!fs.existsSync(localPolicyDir)) fs.mkdirSync(localPolicyDir, { recursive: true });
  const localPolicyPath = path.join(localPolicyDir, `dyna_q_policy.${runId}.${stamp}.json`);

  const meta = { id: runId, mode: 'local', status: 'starting', createdAt: Date.now(), projectId, pid: null, logPath, environmentName, policyOutputPath: localPolicyPath };
  writeRunMeta(runId, meta);

  if (environmentSpec) {
    fs.writeFileSync(path.join(runsDir, `${runId}.environment.json`), JSON.stringify(environmentSpec, null, 2));
  }

  const py = process.env.PYTHON || 'python';
  const env = Object.assign({}, process.env, {
    DYNA_TRAIN_SECONDS: String(trainSeconds),
    DYNA_PLANNING_STEPS: String(planningSteps),
    DYNA_EPISODES: String(episodes),
    TRAINING_ENVIRONMENT_NAME: environmentName,
    TRAINING_ENVIRONMENT_SPEC_PATH: environmentSpec ? path.join(runsDir, `${runId}.environment.json`) : '',
    DYNA_POLICY_OUTPUT_PATH: localPolicyPath,
  });

  const policyPath = path.join(repoRoot, 'trainer', 'models', 'dyna_q_policy.json');
  if (fs.existsSync(policyPath)) {
    const backups = path.join(repoRoot, 'trainer', 'models', 'backups');
    if (!fs.existsSync(backups)) fs.mkdirSync(backups, { recursive: true });
    const backupPath = path.join(backups, `dyna_q_policy.local_pre_train.${stamp}.json`);
    fs.copyFileSync(policyPath, backupPath);
    meta.backupPolicy = backupPath;
    writeRunMeta(runId, meta);
  }

  const outStream = fs.createWriteStream(logPath, { flags: 'a' });
  const child = spawn(py, ['trainer/train_dyna_q.py'], { cwd: repoRoot, env });
  meta.pid = child.pid;
  meta.status = 'running';
  writeRunMeta(runId, meta);

  child.stdout.on('data', (d) => { outStream.write(d); });
  child.stderr.on('data', (d) => { outStream.write(d); });

  child.on('close', (code) => {
    meta.status = code === 0 ? 'complete' : 'error';
    meta.exitCode = code;
    meta.finishedAt = Date.now();
    writeRunMeta(runId, meta);
    outStream.end();
  });

  res.json({ id: runId, logPath, mode: 'local' });
});

app.get('/api/kaggle/runs', (req, res) => {
  const runs = [];
  fs.readdirSync(runsDir).forEach(f => {
    if (f.endsWith('.json')) {
      try {
        const m = JSON.parse(fs.readFileSync(path.join(runsDir, f), 'utf8'));
        if (m && typeof m.id === 'string') runs.push(m);
      } catch(e) {}
    }
  });
  res.json(runs.sort((a,b)=>b.createdAt - a.createdAt));
});

app.get('/api/kaggle/run/:id/log', (req, res) => {
  const id = req.params.id;
  const metaPath = path.join(runsDir, `${id}.json`);
  if (!fs.existsSync(metaPath)) return res.status(404).json({ error: 'run not found' });
  const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
  if (!fs.existsSync(meta.logPath)) return res.json({ log: '' });
  const log = fs.readFileSync(meta.logPath, 'utf8');
  res.json({ log, meta });
});

// Root status page
app.get('/', (req, res) => {
  res.send(`
    <html>
      <head><title>DRL Wildfire Kaggle API</title></head>
      <body>
        <h1>DRL Wildfire Kaggle API</h1>
        <p>Server is running. Useful endpoints:</p>
        <ul>
          <li><a href="/api/kaggle/runs">/api/kaggle/runs</a></li>
          <li><a href="/api/policies">/api/policies</a></li>
          <li><a href="/api/policy/current">/api/policy/current</a></li>
        </ul>
        <p>Use the frontend page or the API to start training runs.</p>
      </body>
    </html>
  `);
});

const port = process.env.PORT || 3500;
app.listen(port, () => console.log(`Kaggle API server listening on http://localhost:${port}`));
