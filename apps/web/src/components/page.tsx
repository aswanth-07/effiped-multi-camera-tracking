import type { ReactNode } from "react";

/** The mono, letter-spaced micro label used above every region on the site. */
export function Label({ children, tone = "note" }: { children: ReactNode; tone?: "note" | "link" | "muted" }) {
  return <p className={`label label--${tone}`}>{children}</p>;
}

export function PageHead({
  index,
  label,
  title,
  lede,
  children
}: {
  index: string;
  label: string;
  title: string;
  lede?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <header className="page-head">
      <Label>
        <span className="page-head__index">{index}</span>
        {label}
      </Label>
      <h1>{title}</h1>
      {lede ? <p className="page-head__lede">{lede}</p> : null}
      {children}
    </header>
  );
}

export function Section({
  id,
  index,
  label,
  title,
  lede,
  children
}: {
  id?: string;
  index?: string;
  label?: string;
  title?: string;
  lede?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="section" id={id}>
      {label ? (
        <Label>
          {index ? <span className="page-head__index">{index}</span> : null}
          {label}
        </Label>
      ) : null}
      {title ? <h2 className="section__title">{title}</h2> : null}
      {lede ? <p className="section__lede">{lede}</p> : null}
      {children}
    </section>
  );
}

/**
 * A figure with the protocol it was measured under. The protocol line is not
 * decoration: a number without it is not a result, and the project's own
 * evidence ledger separates them for the same reason.
 */
export function Metric({
  value,
  unit,
  name,
  protocol,
  tone = "link"
}: {
  value: string;
  unit?: string;
  name: string;
  protocol: string;
  tone?: "link" | "note" | "merged" | "plain";
}) {
  return (
    <article className={`metric metric--${tone}`}>
      <strong>
        {value}
        {unit ? <span className="metric__unit">{unit}</span> : null}
      </strong>
      <span className="metric__name">{name}</span>
      <small className="metric__protocol">{protocol}</small>
    </article>
  );
}

export function Callout({
  tone = "warning",
  title,
  children
}: {
  tone?: "warning" | "note" | "danger";
  title: string;
  children: ReactNode;
}) {
  return (
    <aside className={`callout callout--${tone}`}>
      <strong>{title}</strong>
      <div>{children}</div>
    </aside>
  );
}

export type Column = { key: string; label: string; numeric?: boolean };
export type Row = { key: string; cells: Record<string, ReactNode>; emphasis?: "positive" | "quiet" };

export function DataTable({
  label,
  caption,
  columns,
  rows,
  source
}: {
  /** Names the scroll region. A narrow viewport makes the table scroll
      horizontally, and a scrollable region has to be reachable by keyboard. */
  label: string;
  caption?: ReactNode;
  columns: Column[];
  rows: Row[];
  source?: { label: string; href: string };
}) {
  return (
    <figure className="table-wrap">
      <div aria-label={label} className="table-scroll" role="group" tabIndex={0}>
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((column) => (
                <th className={column.numeric ? "is-num" : undefined} key={column.key} scope="col">
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr className={row.emphasis ? `is-${row.emphasis}` : undefined} key={row.key}>
                {columns.map((column) => (
                  <td className={column.numeric ? "is-num" : undefined} key={column.key}>
                    {row.cells[column.key]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {caption || source ? (
        <figcaption>
          {caption}
          {source ? (
            <>
              {" "}
              <a href={source.href} rel="noreferrer" target="_blank">
                {source.label}
              </a>
            </>
          ) : null}
        </figcaption>
      ) : null}
    </figure>
  );
}

/** A numbered step in an explanation of the pipeline. */
export function Step({
  n,
  title,
  children
}: {
  n: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <article className="step">
      <span className="step__n">{n}</span>
      <div>
        <strong>{title}</strong>
        <p>{children}</p>
      </div>
    </article>
  );
}

export function Code({ children, label = "Code sample" }: { children: string; label?: string }) {
  // A narrow viewport makes this scroll sideways, and a scrollable region has
  // to be reachable by keyboard.
  return (
    <pre aria-label={label} className="code-block" role="group" tabIndex={0}>
      <code>{children}</code>
    </pre>
  );
}
