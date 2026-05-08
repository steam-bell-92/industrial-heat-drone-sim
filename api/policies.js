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

  const repoRoot = path.resolve(__dirname, '..');
  const out = [];
  
  const trainerModels = path.join(repoRoot, 'trainer', 'models');
  if (fs.existsSync(trainerModels)) {
    try {
      fs.readdirSync(trainerModels).forEach(f => {
        if (f.endsWith('.json')) out.push(path.join(trainerModels, f));
      });
    } catch (e) {}
  }
  
  const localRuns = path.join(repoRoot, 'trainer', 'models', 'local_runs');
  if (fs.existsSync(localRuns)) {
    try {
      fs.readdirSync(localRuns).forEach(f => {
        if (f.endsWith('.json')) out.push(path.join(localRuns, f));
      });
    } catch (e) {}
  }
  
  const shared = path.join(repoRoot, 'shared', 'projects');
  if (fs.existsSync(shared)) {
    try {
      fs.readdirSync(shared).forEach(p => {
        const t = path.join(shared, p, 'training', 'dyna_q_policy.json');
        if (fs.existsSync(t)) out.push(t);
      });
    } catch (e) {}
  }
  
  res.json(out);
};
