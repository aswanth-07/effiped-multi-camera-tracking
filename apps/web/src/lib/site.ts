/**
 * Route table and the handful of external URLs the site links to.
 *
 * Every number the interface prints comes from research/results/summary.json
 * through src/data/results.ts. Nothing factual is written here.
 */

export const repository = "https://github.com/aswanth-07/effiped-multi-camera-tracking";
export const boxjdeRepository = "https://github.com/aswanth-07/boxjde-person-search";
export const author = "https://github.com/aswanth-07";

export type RouteId = "overview" | "workbench" | "system" | "evidence" | "deploy";

export type RouteDef = {
  id: RouteId;
  path: string;
  /** Rail label. Short, because the rail is narrow. */
  label: string;
  /** Page title and the document title suffix. */
  title: string;
  /** One line describing the route, used for the meta description. */
  summary: string;
};

export const routes: RouteDef[] = [
  {
    id: "overview",
    path: "/",
    label: "Overview",
    title: "Overview",
    summary:
      "What EffiPed does, what it was measured at, and where the evidence for each figure lives."
  },
  {
    id: "workbench",
    path: "/workbench",
    label: "Workbench",
    title: "Identity review workbench",
    summary:
      "The six-panel review application, running over four precomputed P-DESTRE camera clips."
  },
  {
    id: "system",
    path: "/system",
    label: "System",
    title: "How the system works",
    summary:
      "One backbone feeding detection and a part-aware identity descriptor, then two scales of association."
  },
  {
    id: "evidence",
    path: "/evidence",
    label: "Evidence",
    title: "Evidence and protocols",
    summary:
      "Every published figure with the protocol it was measured under, and what it does not establish."
  },
  {
    id: "deploy",
    path: "/deploy",
    label: "Run it",
    title: "Run it locally",
    summary:
      "Install the package, start the local service, and what the withheld weights mean in practice."
  }
];

export function routeForPath(path: string): RouteDef | undefined {
  return routes.find((route) => route.path === path);
}
