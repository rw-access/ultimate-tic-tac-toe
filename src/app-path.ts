export const APP_BASE_PATH = '/play/ultimate-tic-tac-toe';

export function stripAppBasePath(pathname: string): string | null {
  if (pathname === APP_BASE_PATH) return '/';
  if (pathname.startsWith(`${APP_BASE_PATH}/`)) return pathname.slice(APP_BASE_PATH.length);
  return null;
}

export function appPath(currentPathname: string, path: string): string {
  const suffix = path.startsWith('/') ? path : `/${path}`;
  return `${stripAppBasePath(currentPathname) === null ? '' : APP_BASE_PATH}${suffix}`;
}
