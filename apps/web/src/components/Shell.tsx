import { useEffect, useRef, useState } from "react";
import { Cpu, ExternalLink, Menu, ShieldAlert, X } from "lucide-react";

import { Link, useRouter } from "../lib/router";
import { repository, routes } from "../lib/site";

/** The drawn mark. Square, flat, and the same shape at every size. */
function Mark() {
  return (
    <svg aria-hidden="true" className="mark" viewBox="0 0 48 48">
      <rect height="47" width="47" x="0.5" y="0.5" />
      <path d="M12 33 L20.5 15 L27.5 33" />
      <path d="M15.6 27.4 H24.9" />
      <path d="M31 33 V15 H36 a5 5 0 0 1 0 10 H31" />
      <path d="M34.4 25 L38.5 33" />
    </svg>
  );
}

export function Shell({ children }: { children: React.ReactNode }) {
  const { path } = useRouter();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuButton = useRef<HTMLButtonElement>(null);

  // The drawer is a route-level overlay: leaving the route closes it, and so
  // does Escape, which is where a keyboard user reaches for it first.
  useEffect(() => setMenuOpen(false), [path]);
  useEffect(() => {
    if (!menuOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMenuOpen(false);
        menuButton.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [menuOpen]);

  return (
    <div className="frame">
      <a className="skip-link" href="#main">
        Skip to content
      </a>

      <header className="topbar">
        <Link aria-label="EffiPed home" className="topbar__brand" to="/">
          <Mark />
          <span>EffiPed</span>
        </Link>
        <button
          aria-expanded={menuOpen}
          aria-label={menuOpen ? "Close navigation" : "Open navigation"}
          className="topbar__toggle"
          onClick={() => setMenuOpen((open) => !open)}
          ref={menuButton}
          type="button"
        >
          {menuOpen ? <X aria-hidden="true" size={18} /> : <Menu aria-hidden="true" size={18} />}
        </button>
      </header>

      <nav
        aria-label="Primary"
        className={menuOpen ? "rail is-open" : "rail"}
        id="site-nav"
      >
        <Link aria-label="EffiPed home" className="rail__brand" to="/">
          <Mark />
        </Link>

        <div className="rail__links">
          {routes.map((route) => (
            <Link
              aria-current={path === route.path ? "page" : undefined}
              className="rail__link"
              key={route.id}
              to={route.path}
            >
              <span>{route.label}</span>
            </Link>
          ))}
          <a className="rail__link rail__link--out" href={repository} rel="noreferrer" target="_blank">
            <span>Source</span>
            <ExternalLink aria-hidden="true" size={12} />
          </a>
        </div>

        <div className="rail__context">
          <p>Precomputed replay</p>
          <div>
            <Cpu aria-hidden="true" />
            <span>No inference in the browser</span>
          </div>
          <div>
            <ShieldAlert aria-hidden="true" />
            <span>Ranking, not identification</span>
          </div>
          <small>Software Apache-2.0. P-DESTRE media CC BY-NC-SA 4.0.</small>
        </div>
      </nav>

      {menuOpen ? (
        <button
          aria-hidden="true"
          className="rail-scrim"
          onClick={() => setMenuOpen(false)}
          tabIndex={-1}
          type="button"
        />
      ) : null}

      <main id="main">{children}</main>

      <footer className="site-foot">
        <div className="site-foot__row">
          <div>
            <strong>EffiPed</strong>
            <span>Multi-camera pedestrian detection, tracking and identity review</span>
          </div>
          <div className="site-foot__links">
            <a href={repository} rel="noreferrer" target="_blank">
              Repository
            </a>
            <a href="/report/effiped-technical-report.pdf" rel="noreferrer" target="_blank">
              Technical report
            </a>
            <a href="/media/ASSET_MANIFEST.json" rel="noreferrer" target="_blank">
              Media attribution
            </a>
          </div>
        </div>
        <p className="site-foot__legal">
          Software &copy; 2026 Aswanth Raj under Apache-2.0. P-DESTRE-derived media is separately
          licensed CC BY-NC-SA 4.0 for this non-commercial research demonstration. Ranked matches
          are reviewable appearance evidence, not proof of identity.
        </p>
      </footer>
    </div>
  );
}
