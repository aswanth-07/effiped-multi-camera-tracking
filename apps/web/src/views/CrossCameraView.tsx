import type { RunResult, Settings } from "../engine/pipeline";
import { sessionStats } from "../data/personSearch";

export function CrossCameraView({
  result,
  settings,
  onInspect
}: {
  result: RunResult | null;
  settings: Settings;
  onInspect: (personId: string) => void;
}) {
  if (!result || result.sources.length === 0) {
    return <div className="empty"><p>Run the pipeline to build cross-camera links.</p></div>;
  }

  const ids = result.sources.map((source) => source.id);

  // Live link counts at the current similarity floor, per ordered camera pair.
  const counts = new Map<string, number>();
  for (const row of result.people) {
    for (const match of row.matches) {
      if (match.same_video) continue;
      const key = `${row.person.video_index}:${match.person.video_index}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }

  const strongest = [...result.people]
    .flatMap((row) => row.matches.filter((m) => !m.same_video).map((match) => ({ row, match })))
    .sort((a, b) => b.match.similarity - a.match.similarity)
    .slice(0, 12);

  return (
    <div className="view view--cross">
      <div className="view__head">
        <h2>Cross camera association</h2>
        <span className="view__count">
          {result.stats.crossLinks} links above {settings.similarity.toFixed(2)}
        </span>
      </div>

      <div className="split split--matrix">
        <div>
          <p className="panel-label">Link matrix, query camera to gallery camera</p>
          <table className="matrix">
            <thead>
              <tr>
                <th scope="col"><span className="sr-only">Query camera</span></th>
                {ids.map((id) => (
                  <th key={id} scope="col">C{id + 1}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ids.map((rowId) => (
                <tr key={rowId}>
                  <th scope="row">C{rowId + 1}</th>
                  {ids.map((colId) => {
                    const value = rowId === colId ? null : counts.get(`${rowId}:${colId}`) ?? 0;
                    const intensity = value ? Math.min(1, value / 12) : 0;
                    return (
                      <td
                        className={rowId === colId ? "matrix__self" : "matrix__cell"}
                        key={colId}
                        style={value ? { background: `rgba(89, 165, 252, ${0.08 + intensity * 0.5})` } : undefined}
                      >
                        {rowId === colId ? "." : value}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="view__foot">
            Recomputed from the stored pairwise similarities every time a threshold moves.
          </p>
        </div>

        <div>
          <p className="panel-label">Strongest links at this threshold</p>
          <ol className="linklist">
            {strongest.map(({ row, match }) => (
              <li key={`${row.person.id}-${match.person.id}`}>
                <button onClick={() => onInspect(row.person.id)} type="button">
                  <span className="linklist__pair">
                    <code>v{row.person.video_index + 1}/t{row.person.track_id}</code>
                    <span aria-hidden="true">&rarr;</span>
                    <code>v{match.person.video_index + 1}/t{match.person.track_id}</code>
                  </span>
                  <span className="linklist__score">{match.similarity.toFixed(3)}</span>
                </button>
              </li>
            ))}
            {strongest.length === 0 ? <li className="linklist__none">No link clears this threshold.</li> : null}
          </ol>
        </div>
      </div>

      <div className="archive">
        <p className="panel-label">Archived session diagnostic</p>
        <table className="grid-table">
          <thead>
            <tr>
              <th scope="col">Pair</th>
              <th className="is-num" scope="col">Correct</th>
              <th className="is-num" scope="col">Wrong</th>
              <th className="is-num" scope="col">Shared</th>
              <th className="is-num" scope="col">Precision</th>
            </tr>
          </thead>
          <tbody>
            {sessionStats.perPair.map((pair) => (
              <tr className={pair.precision >= 0.99 ? "is-good" : pair.precision < 0.6 ? "is-weak" : undefined} key={pair.pair}>
                <th scope="row">{pair.pair}</th>
                <td className="is-num">{pair.correct}</td>
                <td className="is-num">{pair.wrong}</td>
                <td className="is-num">{pair.shared}</td>
                <td className="is-num">{pair.precision.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="view__foot">
          {sessionStats.crossCameraIds} cross-camera identities over {sessionStats.maxFrames} frames,
          at {sessionStats.precision.toFixed(2)} overall pairwise precision. Cameras 1 and 2 associate
          perfectly; cameras 3 and 4 fall to 0.54. This is an archived application diagnostic from the
          original run, not a benchmark, and it does not move with the sliders.
        </p>
      </div>
    </div>
  );
}
