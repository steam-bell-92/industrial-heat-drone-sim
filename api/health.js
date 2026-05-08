export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader('Access-Control-Allow-Headers', 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version');
  
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  res.status(200).json({
    status: 'ok',
    message: 'Industrial Heat Drone API is running',
    timestamp: new Date().toISOString(),
    endpoints: {
      policies: '/api/policies',
      policy_current: '/api/policy/current',
      policies_content: '/api/policies/content',
      policies_activate: '/api/policies/activate',
      kaggle_start: '/api/kaggle/start',
      kaggle_runs: '/api/kaggle/runs',
      kaggle_run_log: '/api/kaggle/run/:id/log'
    }
  });
}
