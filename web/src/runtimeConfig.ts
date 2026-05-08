const DEFAULT_TRAINING_API_URL = 'http://localhost:3500';
const DEFAULT_BLUEPRINT_API_URL = 'http://localhost:5001';

function readConfiguredUrl(envKey: string): string | undefined {
  const env = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env ?? {};
  const value = env[envKey]?.trim();
  return value && value.length > 0 ? value.replace(/\/$/, '') : undefined;
}

function getBrowserOriginFallback(defaultUrl: string): string {
  if (typeof window === 'undefined') {
    return defaultUrl;
  }

  const { origin } = window.location;
  const isLocalhost = origin.includes('localhost') || origin.includes('127.0.0.1');
  return isLocalhost ? defaultUrl : origin.replace(/\/$/, '');
}

export function getTrainingApiBaseUrl(): string {
  return readConfiguredUrl('VITE_API_BASE_URL') ?? getBrowserOriginFallback(DEFAULT_TRAINING_API_URL);
}

export function getBlueprintApiBaseUrl(): string {
  return (
    readConfiguredUrl('VITE_BLUEPRINT_API_URL') ??
    readConfiguredUrl('VITE_API_BASE_URL') ??
    getBrowserOriginFallback(DEFAULT_BLUEPRINT_API_URL)
  );
}