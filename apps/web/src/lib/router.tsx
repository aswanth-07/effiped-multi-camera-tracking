/**
 * A very small History-API router.
 *
 * The site is four content routes plus the workbench, with no nested layouts,
 * no loaders and no code splitting beyond what Vite already does. A routing
 * library would be more dependency surface than the problem deserves, and this
 * application's dependency graph is audited in CI.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type AnchorHTMLAttributes,
  type ReactNode
} from "react";

type RouterValue = {
  path: string;
  navigate: (to: string) => void;
};

const RouterContext = createContext<RouterValue>({
  path: "/",
  navigate: () => {}
});

function currentPath(): string {
  if (typeof window === "undefined") return "/";
  const raw = window.location.pathname || "/";
  // Treat "/system/" and "/system" as the same route.
  return raw.length > 1 && raw.endsWith("/") ? raw.slice(0, -1) : raw;
}

export function Router({ children }: { children: ReactNode }) {
  const [path, setPath] = useState(currentPath);

  useEffect(() => {
    const onPop = () => setPath(currentPath());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const navigate = useCallback((to: string) => {
    if (to === currentPath()) return;
    window.history.pushState({}, "", to);
    setPath(currentPath());
    // A route change is a new page, so the reader starts at its top. Respect a
    // reduced-motion preference rather than animating the jump.
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" });
  }, []);

  const value = useMemo(() => ({ path, navigate }), [path, navigate]);
  return <RouterContext.Provider value={value}>{children}</RouterContext.Provider>;
}

export function useRouter() {
  return useContext(RouterContext);
}

type LinkProps = AnchorHTMLAttributes<HTMLAnchorElement> & { to: string };

/**
 * Renders a real anchor so middle-click, modifier-click and "copy link address"
 * all behave, and only takes over the plain left click.
 */
export function Link({ to, onClick, children, ...rest }: LinkProps) {
  const { navigate } = useRouter();
  return (
    <a
      href={to}
      onClick={(event) => {
        onClick?.(event);
        if (event.defaultPrevented) return;
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        if (event.button !== 0) return;
        event.preventDefault();
        navigate(to);
      }}
      {...rest}
    >
      {children}
    </a>
  );
}
