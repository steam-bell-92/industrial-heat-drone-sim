const fs = require('fs');
const path = require('path');

export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader('Access-Control-Allow-Headers', 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version');
  
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  const repoRoot = path.resolve(__dirname, '..');
  const policyPath = path.join(repoRoot, 'trainer', 'models', 'dyna_q_policy.json');
  
  if (!fs.existsSync(policyPath)) {
    return res.json({ exists: false });
  }
  
  try {
    const st = fs.statSync(policyPath);
    return res.json({ exists: true, path: policyPath, mtime: st.mtimeMs, size: st.size });
  } catch (err) {
    return res.status(500).json({ error: 'Failed to read policy' });
  }
}
