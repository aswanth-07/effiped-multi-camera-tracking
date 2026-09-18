import { useEffect } from "react";

import { Shell } from "./components/Shell";
import { Link, useRouter } from "./lib/router";
import { routeForPath } from "./lib/site";
import { Deploy } from "./pages/Deploy";
import { Evidence } from "./pages/Evidence";
import { Overview } from "./pages/Overview";
import { System } from "./pages/System";
import { WorkbenchPage } from "./pages/WorkbenchPage";

function NotFound({ path }: { path: string }) {
  return (
    <section className="not-found">
      <p className="label label--note">404</p>
      <h1>No page at {path}</h1>
      <p>
        That route does not exist on this site. The overview lists everything that does.
      </p>
      <Link className="primary-link" to="/">
        Back to the overview
      </Link>
    </section>
  );
}

export default function App() {
  const { path } = useRouter();
  const route = routeForPath(path);

  useEffect(() => {
    const suffix = "EffiPed";
    document.title = route ? `${route.title} | ${suffix}` : `Not found | ${suffix}`;
    const description = document.querySelector('meta[name="description"]');
    if (description && route) description.setAttribute("content", route.summary);
  }, [route]);

  return (
    <Shell>
      {route?.id === "overview" ? <Overview /> : null}
      {route?.id === "workbench" ? <WorkbenchPage /> : null}
      {route?.id === "system" ? <System /> : null}
      {route?.id === "evidence" ? <Evidence /> : null}
      {route?.id === "deploy" ? <Deploy /> : null}
      {route ? null : <NotFound path={path} />}
    </Shell>
  );
}
