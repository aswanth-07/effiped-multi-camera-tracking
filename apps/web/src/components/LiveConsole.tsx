import { useEffect, useMemo, useRef, useState } from "react";
import { Camera, LoaderCircle, Play, Search, Trash2, UploadCloud } from "lucide-react";

type Model = {
  key: string;
  label: string;
  description: string;
  descriptor_dim: number;
  artifact_version: string;
  available: boolean;
  benchmark: Record<string, string | number>;
};

type Job = {
  job_id: string;
  status: string;
  message: string;
  progress: number;
  people_count: number;
};

type Person = {
  id: string;
  video_index: number;
  track_id: number;
  num_samples: number;
  best_score: number;
  best_crop_asset?: string;
  caption: string;
};

type Match = { person: Person; similarity: number; same_video: boolean };

const apiBase = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

function assetUrl(asset?: string) {
  if (!asset) return "";
  return `${apiBase}/api/assets/${asset}`;
}

export function LiveConsole() {
  const [models, setModels] = useState<Model[]>([]);
  const [modelKey, setModelKey] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [people, setPeople] = useState<Person[]>([]);
  const [selected, setSelected] = useState<Person | null>(null);
  const [matches, setMatches] = useState<Match[]>([]);
  const [error, setError] = useState("");
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    fetch(`${apiBase}/api/models`)
      .then((response) => {
        if (!response.ok) throw new Error("Local API is unavailable.");
        return response.json();
      })
      .then((rows: Model[]) => {
        setModels(rows);
        setModelKey(rows.find((row) => row.available)?.key ?? rows[0]?.key ?? "");
      })
      .catch((reason) => setError(String(reason.message ?? reason)));
  }, []);

  const selectedModel = useMemo(() => models.find((model) => model.key === modelKey), [models, modelKey]);

  async function loadPeople(jobId: string) {
    const response = await fetch(`${apiBase}/api/person-search/jobs/${jobId}/people`);
    if (!response.ok) throw new Error("Could not load indexed people.");
    const payload = await response.json();
    setPeople(payload.people ?? []);
  }

  async function startJob() {
    if (!files.length || !modelKey) return;
    setError("");
    setPeople([]);
    setMatches([]);
    setSelected(null);
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    form.append("model_key", modelKey);
    const response = await fetch(`${apiBase}/api/person-search/jobs`, { method: "POST", body: form });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail ?? "Could not create the indexing job.");
    }
    const created = await response.json();
    setJob({ job_id: created.job_id, status: "queued", message: "Queued", progress: 0, people_count: 0 });

    const wsRoot = apiBase
      ? apiBase.replace(/^http/, "ws")
      : `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}`;
    const socket = new WebSocket(`${wsRoot}/api/person-search/jobs/${created.job_id}/stream`);
    socketRef.current = socket;
    socket.onmessage = async (event) => {
      const message = JSON.parse(event.data);
      const status = message.payload?.job;
      if (status) setJob(status);
      if (message.type === "job_complete") {
        await loadPeople(created.job_id);
      }
      if (message.type === "error") setError(message.payload?.message ?? "Indexing failed.");
    };
  }

  async function inspectPerson(person: Person) {
    if (!job) return;
    setSelected(person);
    const response = await fetch(`${apiBase}/api/person-search/jobs/${job.job_id}/people/${person.id}/matches`);
    if (!response.ok) return;
    setMatches(await response.json());
  }

  async function deleteJob() {
    if (!job) return;
    socketRef.current?.close();
    const response = await fetch(`${apiBase}/api/person-search/jobs/${job.job_id}`, { method: "DELETE" });
    if (!response.ok) {
      setError("The job is still running and cannot be deleted yet.");
      return;
    }
    setJob(null);
    setPeople([]);
    setMatches([]);
    setSelected(null);
    setFiles([]);
  }

  return (
    <div className="live-console">
      <header className="live-console__header">
        <div>
          <span className="label">Local CUDA identity review</span>
          <h3>Index authorized camera video on this machine.</h3>
        </div>
        <span className="live-console__state"><span /> API-backed mode</span>
      </header>

      <div className="live-console__setup">
        <label className="upload-zone">
          <UploadCloud />
          <strong>{files.length ? `${files.length} camera file${files.length > 1 ? "s" : ""} selected` : "Select up to four videos"}</strong>
          <span>MP4, MOV, AVI, MKV, or WebM · processed locally</span>
          <input
            type="file"
            accept="video/*"
            multiple
            onChange={(event) => setFiles(Array.from(event.target.files ?? []).slice(0, 4))}
          />
        </label>
        <div className="live-console__controls">
          <label>
            Model manifest
            <select value={modelKey} onChange={(event) => setModelKey(event.target.value)}>
              {models.map((model) => (
                <option value={model.key} key={model.key}>
                  {model.label}{model.available ? "" : " — checkpoint unavailable"}
                </option>
              ))}
            </select>
          </label>
          {selectedModel && (
            <p>{selectedModel.descriptor_dim}-D · artifact {selectedModel.artifact_version} · {selectedModel.description}</p>
          )}
          <button
            className="button button--primary"
            disabled={!files.length || !selectedModel?.available || Boolean(job && ["queued", "running"].includes(job.status))}
            onClick={() => startJob().catch((reason) => setError(String(reason.message ?? reason)))}
          >
            {job && ["queued", "running"].includes(job.status) ? <LoaderCircle className="spin" /> : <Play />}
            Start local indexing
          </button>
        </div>
      </div>

      {error && <p className="live-console__error">{error}</p>}
      {job && (
        <div className="job-strip">
          <div><span>Job {job.job_id}</span><strong>{job.message}</strong></div>
          <div className="progress"><i style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>
          <span>{Math.round(job.progress * 100)}% · {job.people_count} people</span>
          <button onClick={deleteJob} title="Delete job and generated assets"><Trash2 /></button>
        </div>
      )}

      {people.length > 0 && (
        <div className="review-grid">
          <div>
            <div className="review-grid__title"><Camera /> Indexed gallery <span>{people.length}</span></div>
            <div className="people-grid">
              {people.map((person) => (
                <button className={selected?.id === person.id ? "selected" : ""} onClick={() => inspectPerson(person)} key={person.id}>
                  {person.best_crop_asset ? <img src={assetUrl(person.best_crop_asset)} alt="" /> : <Camera />}
                  <strong>{person.id}</strong>
                  <span>Cam {person.video_index + 1} · {person.num_samples} views</span>
                </button>
              ))}
            </div>
          </div>
          <aside>
            <div className="review-grid__title"><Search /> Cross-camera candidates</div>
            {selected ? matches.map((match) => (
              <article key={match.person.id}>
                {match.person.best_crop_asset && <img src={assetUrl(match.person.best_crop_asset)} alt="" />}
                <div><strong>{match.person.id}</strong><span>{Math.round(match.similarity * 100)}% similarity</span></div>
              </article>
            )) : <p>Select a gallery track to rank identity evidence.</p>}
          </aside>
        </div>
      )}
      <p className="candidate-panel__note">Uploaded video, crops, descriptors, and rankings are sensitive. Delete the job after review.</p>
    </div>
  );
}
