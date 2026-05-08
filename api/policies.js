const fs = require('fs');
const path = require('path');

module.exports = function handler(req, res) {
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader('Access-Control-Allow-Headers', 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version');
  
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  try {
    const repoRoot = path.resolve(__dirname, '..');
    const out = [];
    
    const trainerModels = path.join(repoRoot, 'trainer', 'models');
    if (fs.existsSync(trainerModels)) {
      try {
        fs.readdirSync(trainerModels).forEach(f => {
          if (f.endsWith('.json')) out.push(path.join(trainerModels, f));
        });
      } catch (e) {
        console.warn('[api/policies] Error reading trainer/models:', e);
      }
    }
    
    const localRuns = path.join(repoRoot, 'trainer', 'models', 'local_runs');
    if (fs.existsSync(localRuns)) {
      try {
        fs.readdirSync(localRuns).forEach(f => {
          if (f.endsWith('.json')) out.push(path.join(localRuns, f));
        });
      } catch (e) {
        console.warn('[api/policies] Error reading local_runs:', e);
      }
    }
    
    const shared = path.join(repoRoot, 'shared', 'projects');
    if (fs.existsSync(shared)) {
      try {
        fs.readdirSync(shared).forEach(p => {
          const t = path.join(shared, p, 'training', 'dyna_q_policy.json');
          if (fs.existsSync(t)) out.push(t);
        });
      } catch (e) {
        console.warn('[api/policies] Error reading shared projects:', e);
      }
    }
    
    console.log('[api/policies] Found policies:', out);
    res.json(out);
  } catch (e) {
    console.error('[api/policies] Error:', e);
    res.status(500).json({ error: 'Failed to list policies' });
  }
};
