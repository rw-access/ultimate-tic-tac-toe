export const APP_BASE_PATH = '/play/ultimate-tic-tac-toe';
const LEGACY_APP_BASE_PATH = '/ultimate-tic-tac-toe';

function matchingBasePath(pathname: string): string | undefined {
  return [APP_BASE_PATH, LEGACY_APP_BASE_PATH].find(base =>
    pathname === base || pathname.startsWith(`${base}/`));
}

export function stripAppBasePath(pathname: string): string | null {
  const base = matchingBasePath(pathname);
  return base === undefined ? null : pathname.slice(base.length) || '/';
}

export function appPath(currentPathname: string, path: string): string {
  const suffix = path.startsWith('/') ? path : `/${path}`;
  return `${matchingBasePath(currentPathname) ?? ''}${suffix}`;
}
