# DRL Wildfire

This repository contains the autonomous drone heat-zone detection project, including the Vite web app, the Node.js training API, and the reinforcement-learning training scripts.

## Deployment Model

- Frontend: Vercel
- Source control and CI: GitHub
- API host: configurable through environment variables

## Vercel Deployment

**Live Deployment:**
- Production: https://heatdrone.vercel.app
- Full URL: https://industrial-heat-drone-mvd0sepyu-anshu-bhawsars-projects.vercel.app

The root `vercel.json` is configured to build the `web` project and serve the Vite output.

### Environment Variables

Set these in Vercel project settings → Environment Variables:

- `VITE_API_BASE_URL` - backend training API base URL (e.g., `https://your-backend-domain.com`)
- `VITE_BLUEPRINT_API_URL` - blueprint processing API base URL

If you host the backend on the same domain or proxy through Vercel, the app will automatically use the current origin outside localhost. For a separate backend, set the environment variables to your API endpoint.

## GitHub Workflow

The repository includes a GitHub Actions workflow at `.github/workflows/ci.yml` that installs the web dependencies and runs a production build on every push and pull request.

## Backend API Deployment

The `server/index.js` Express API provides training endpoints (`/api/local/start`, `/api/kaggle/start`, `/api/policies`, etc.). This is currently a separate Node.js service that must be deployed independently:

**Deployment Options:**
- **Local development:** `npm start` in the `server` directory (runs on port 3500)
- **Cloud hosting:** Render, Railway, Heroku, AWS EC2, or similar Node.js-capable platforms
- **Vercel API Routes:** (Advanced) Refactor `server/index.js` into `/api/*.js` serverless functions

For production, after deploying the backend, set the `VITE_API_BASE_URL` environment variable in Vercel to your backend's public URL.

## Local Development

Frontend:

```powershell
Set-Location web
npm install
npm run dev
```

Backend:

```powershell
Set-Location server
npm install
npm start
```

## Notes

- The frontend no longer hardcodes localhost for API access.
- The deployment configuration keeps the project compatible with GitHub-based workflows and Vercel hosting.