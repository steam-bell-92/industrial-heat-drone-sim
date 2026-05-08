# DRL Wildfire

This repository contains the autonomous drone heat-zone detection project, including the Vite web app, the Node.js training API, and the reinforcement-learning training scripts.

## Deployment Model

- Frontend: Vercel
- Source control and CI: GitHub
- API host: configurable through environment variables

## Vercel Setup

The root `vercel.json` is configured to build the `web` project and serve the Vite output.

Recommended environment variables in Vercel:

- `VITE_API_BASE_URL` - backend training API base URL
- `VITE_BLUEPRINT_API_URL` - blueprint processing API base URL

If you deploy the backend separately, point both variables to that service. If you proxy the backend through the same Vercel domain, the app will fall back to the current origin outside localhost.

## GitHub Workflow

The repository includes a GitHub Actions workflow at `.github/workflows/ci.yml` that installs the web dependencies and runs a production build on every push and pull request.

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