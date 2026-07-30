import {
  Activity,
  BriefcaseBusiness,
  CheckCircle2,
  ChevronsLeft,
  ChevronsRight,
  Crosshair,
  Database,
  Expand,
  FileSearch,
  Gauge,
  Images,
  Pause,
  Play,
  Route,
  Search,
  ShieldCheck,
  UserRound,
  Video
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { results } from "../data/results";
import type { DemoSubject } from "../types";

type WorkspaceTab = "index" | "investigation";

type TimelineSegment = {
  start: number;
  duration: number;
  color: string;
};

const timelineSegments: TimelineSegment[][] = [
  [
    { start: 0.5, duration: 2.5, color: "#22c7b8" },
    { start: 4.2, duration: 4.6, color: "#7dd3fc" },
    { start: 10.6, duration: 2.8, color: "#f59e0b" }
  ],
  [
    { start: 1.4, duration: 3.1, color: "#a78bfa" },
    { start: 5.3, duration: 5.4, color: "#22c7b8" },
    { start: 11.6, duration: 2.1, color: "#34d399" }
  ],
  [
    { start: 0.2, duration: 4.8, color: "#f472b6" },
    { start: 6.1, duration: 3.7, color: "#60a5fa" },
    { start: 10.5, duration: 4.0, color: "#22c7b8" }
  ],
  [
    { start: 2.2, duration: 3.2, color: "#facc15" },
    { start: 6.5, duration: 5.3, color: "#22c7b8" },
    { start: 12.2, duration: 2.2, color: "#c084fc" }
  ]
];

const candidateSeekTimes = [4.2, 6.1, 8.4, 10.8];

function formatClock(seconds: number) {
  const value = Math.max(0, Math.floor(seconds));
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, "0")}`;
}

function MetricCard({
  icon: Icon,
  label,
  value
}: {
  icon: typeof Video;
  label: string;
  value: string;
}) {
  return (
    <div className="metric-card">
      <Icon aria-hidden="true" size={17} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SubjectPicker({
  subjects,
  selectedId,
  onSelect
}: {
  subjects: DemoSubject[];
  selectedId: string;
  onSelect: (subject: DemoSubject) => void;
}) {
  return (
    <div className="subject-picker" aria-label="Indexed subject tracks">
      {subjects.map((subject) => (
        <button
          className={subject.id === selectedId ? "selected" : ""}
          key={subject.id}
          onClick={() => onSelect(subject)}
          type="button"
        >
          <img src={subject.query_image} alt={`${subject.label} query crop`} />
          <span>ID {subject.id}</span>
          <small>{subject.query_source}</small>
        </button>
      ))}
    </div>
  );
}

export function DemoConsole() {
  const demo = results.demo_case;
  const stats = demo.session_diagnostic;
  const [activeTab, setActiveTab] = useState<WorkspaceTab>("investigation");
  const [selectedSubjectId, setSelectedSubjectId] = useState(demo.subjects[0].id);
  const [selectedVideoId, setSelectedVideoId] = useState(demo.videos[0].id);
  const [selectedCandidateRank, setSelectedCandidateRank] = useState(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(15);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const videoFrameRef = useRef<HTMLDivElement | null>(null);

  const selectedSubject = useMemo(
    () => demo.subjects.find((subject) => subject.id === selectedSubjectId) ?? demo.subjects[0],
    [demo.subjects, selectedSubjectId]
  );
  const selectedVideo = useMemo(
    () => demo.videos.find((video) => video.id === selectedVideoId) ?? demo.videos[0],
    [demo.videos, selectedVideoId]
  );

  useEffect(() => {
    const video = videoRef.current;
    if (video) video.playbackRate = playbackRate;
  }, [playbackRate, selectedVideoId]);

  function selectSubject(subject: DemoSubject) {
    setSelectedSubjectId(subject.id);
    setSelectedCandidateRank(1);
  }

  function seekTo(value: number) {
    const video = videoRef.current;
    const clamped = Math.max(0, Math.min(value, duration || 15));
    if (video) video.currentTime = clamped;
    setCurrentTime(clamped);
  }

  function stepFrame(direction: -1 | 1) {
    seekTo(currentTime + direction / stats.playback_fps);
  }

  function togglePlayback() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) void video.play();
    else video.pause();
  }

  function openCandidate(rank: number) {
    setSelectedCandidateRank(rank);
    setSelectedVideoId("cross-camera");
    window.requestAnimationFrame(() => seekTo(candidateSeekTimes[rank - 1] ?? 4.2));
  }

  return (
    <div className="console-shell" id="demo">
      <section className="topbar">
        <div className="title-block">
          <div className="eyebrow">Video intelligence workbench</div>
          <h2>Identity Review Console</h2>
          <p>
            Explore the original PedestrianTracker replay, inspect indexed subject crops,
            and review ranked cross-camera evidence without downloading a model.
          </p>
        </div>
        <div className="top-metrics">
          <MetricCard icon={Video} label="Sources" value={`${stats.cameras}/4`} />
          <MetricCard icon={Route} label="Local tracks" value={String(stats.local_tracks)} />
          <MetricCard icon={UserRound} label="Global IDs" value={String(stats.cross_camera_ids)} />
          <MetricCard icon={Activity} label="Replay" value={`${stats.playback_fps} FPS`} />
        </div>
      </section>

      <section className="case-strip" aria-label="Analysis workflow" tabIndex={0}>
        <div className="case-summary">
          <BriefcaseBusiness aria-hidden="true" size={16} />
          <div><span>Workspace</span><strong>Four-camera identity review</strong></div>
        </div>
        <div className="case-summary">
          <Database aria-hidden="true" size={16} />
          <div><span>Session</span><strong>{demo.id}</strong></div>
        </div>
        <div className="workflow-step done"><span>1</span>Source intake</div>
        <div className="workflow-step done"><span>2</span>Identity index</div>
        <div className="workflow-step active"><span>3</span>Evidence review</div>
      </section>

      <section className="workspace-tabs" aria-label="Demo workspace tabs">
        <button
          className={activeTab === "index" ? "active" : ""}
          onClick={() => setActiveTab("index")}
          type="button"
        >
          Identity Index
        </button>
        <button
          className={activeTab === "investigation" ? "active" : ""}
          onClick={() => setActiveTab("investigation")}
          type="button"
        >
          Investigation
        </button>
      </section>

      {activeTab === "index" ? (
        <section className="index-layout">
          <aside className="console-panel intake-panel">
            <div className="section-title">
              <ShieldCheck aria-hidden="true" size={18} />
              <h3>Evidence Sources</h3>
            </div>
            <div className="source-grid">
              {[1, 2, 3, 4].map((camera) => (
                <div key={camera}>
                  <CheckCircle2 aria-hidden="true" size={17} />
                  <span>Camera {camera}</span>
                  <small>Indexed replay</small>
                </div>
              ))}
            </div>
            <div className="profile-card">
              <span>Analysis profile</span>
              <strong>EffiPed Tier-1</strong>
              <small>ConvNeXt V2 · CenterNet · part descriptor · BoT-SORT</small>
            </div>
            <button className="primary-action" onClick={() => setActiveTab("investigation")} type="button">
              <Play aria-hidden="true" size={17} />
              Open investigation
            </button>
          </aside>

          <section className="console-panel index-workspace">
            <div className="section-title">
              <UserRound aria-hidden="true" size={18} />
              <h3>Subject Tracks</h3>
              <span>{demo.subjects.length} examples</span>
            </div>
            <SubjectPicker
              subjects={demo.subjects}
              selectedId={selectedSubject.id}
              onSelect={selectSubject}
            />
            <div className="diagnostic-grid">
              <div><span>Processed frames</span><strong>{stats.frames}</strong></div>
              <div><span>Local tracks</span><strong>{stats.local_tracks}</strong></div>
              <div><span>Cross-camera IDs</span><strong>{stats.cross_camera_ids}</strong></div>
              <div><span>Association precision</span><strong>{Math.round(stats.pairwise_association_precision * 100)}%</strong></div>
            </div>
            <p className="diagnostic-note">{stats.label}</p>
          </section>

          <aside className="console-panel detail-panel-static">
            <div className="section-title">
              <Images aria-hidden="true" size={18} />
              <h3>Evidence Review</h3>
            </div>
            <div className="selected-summary">
              <img src={selectedSubject.query_image} alt={`${selectedSubject.label} selected query`} />
              <div>
                <strong>ID {selectedSubject.id}</strong>
                <span>{selectedSubject.query_source} → {selectedSubject.gallery_source}</span>
              </div>
            </div>
            <div className="track-result-list">
              {selectedSubject.candidates.slice(0, 3).map((candidate) => (
                <button
                  className="track-result-card match-high"
                  key={candidate.rank}
                  onClick={() => {
                    openCandidate(candidate.rank);
                    setActiveTab("investigation");
                  }}
                  type="button"
                >
                  <span className="score-badge">{candidate.score.toFixed(3)}</span>
                  <img src={candidate.image} alt={`Rank ${candidate.rank} candidate`} />
                  <div><strong>Rank {candidate.rank}</strong><span>{selectedSubject.gallery_source}</span></div>
                </button>
              ))}
            </div>
          </aside>
        </section>
      ) : (
        <section className="investigation-layout">
          <aside className="investigation-panel query-panel">
            <div className="section-title">
              <Crosshair aria-hidden="true" size={18} />
              <h3>Query Example</h3>
            </div>
            <div className="query-card">
              <img src={selectedSubject.query_image} alt={`${selectedSubject.label} query`} />
              <div className="query-meta">
                <strong>ID {selectedSubject.id} · {selectedSubject.label}</strong>
                <span>Query source {selectedSubject.query_source}</span>
                <span>Gallery source {selectedSubject.gallery_source}</span>
              </div>
            </div>
            <div className="subsection-label">Indexed examples</div>
            <SubjectPicker
              subjects={demo.subjects}
              selectedId={selectedSubject.id}
              onSelect={selectSubject}
            />
            <div className="search-controls">
              <label className="field">
                <span>Search scope</span>
                <select defaultValue="all">
                  <option value="all">Across all camera views</option>
                  <option value="matched">Cross-camera tracks only</option>
                </select>
              </label>
              <div className="camera-checks" aria-label="Camera scope">
                {[1, 2, 3, 4].map((camera) => (
                  <label key={camera}>
                    <input defaultChecked type="checkbox" />
                    <span>C{camera}</span>
                  </label>
                ))}
              </div>
              <button
                className="primary-action compact-action"
                onClick={() => setSelectedVideoId("cross-camera")}
                type="button"
              >
                <Search aria-hidden="true" size={16} />
                Show ranked candidates
              </button>
            </div>
          </aside>

          <section className="investigation-main">
            <div className="investigation-toolbar">
              <div>
                <span>Video playback</span>
                <strong>{selectedVideo.label}</strong>
              </div>
              <select
                aria-label="Replay view"
                value={selectedVideo.id}
                onChange={(event) => {
                  setSelectedVideoId(event.target.value);
                  setCurrentTime(0);
                  setIsPlaying(false);
                }}
              >
                {demo.videos.map((video) => (
                  <option key={video.id} value={video.id}>{video.label}</option>
                ))}
              </select>
            </div>

            <div className="video-stage">
              <div className="video-frame" ref={videoFrameRef}>
                <div className="video-canvas">
                  <video
                    aria-label={selectedVideo.label}
                    key={selectedVideo.source}
                    onLoadedMetadata={(event) => {
                      setDuration(event.currentTarget.duration || 15);
                      event.currentTarget.playbackRate = playbackRate;
                    }}
                    onPause={() => setIsPlaying(false)}
                    onPlay={() => setIsPlaying(true)}
                    onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
                    playsInline
                    poster={selectedVideo.poster}
                    preload="metadata"
                    ref={videoRef}
                    src={selectedVideo.source}
                  />
                  <div className="source-overlay-note">
                    <span className="status-dot" />
                    Original tracker-rendered overlays
                  </div>
                </div>
              </div>
            </div>

            <div className="player-controls">
              <button aria-label="Previous frame" onClick={() => stepFrame(-1)} type="button"><ChevronsLeft size={16} /></button>
              <button aria-label={isPlaying ? "Pause replay" : "Play replay"} onClick={togglePlayback} type="button">
                {isPlaying ? <Pause size={16} /> : <Play size={16} />}
              </button>
              <button aria-label="Next frame" onClick={() => stepFrame(1)} type="button"><ChevronsRight size={16} /></button>
              <span>{formatClock(currentTime)} / {formatClock(duration)}</span>
              <input
                aria-label="Replay timeline"
                max={duration}
                min={0}
                onChange={(event) => seekTo(Number(event.target.value))}
                step={1 / stats.playback_fps}
                type="range"
                value={currentTime}
              />
              <select
                aria-label="Playback speed"
                onChange={(event) => setPlaybackRate(Number(event.target.value))}
                value={playbackRate}
              >
                {[0.5, 1, 1.5, 2].map((rate) => <option key={rate} value={rate}>{rate}×</option>)}
              </select>
              <button
                aria-label="Open fullscreen replay"
                onClick={() => videoFrameRef.current?.requestFullscreen?.()}
                type="button"
              >
                <Expand size={16} />
              </button>
            </div>

            <section className="timeline-panel">
              <div className="timeline-header">
                <div>
                  <span>Detection Timeline</span>
                  <strong>Camera-local track segments</strong>
                </div>
                <span className="timeline-legend"><span /> click a segment to seek</span>
              </div>
              <div className="timeline-lanes">
                {timelineSegments.map((segments, cameraIndex) => (
                  <div className="timeline-lane" key={`camera-${cameraIndex + 1}`}>
                    <span className="lane-label">C{cameraIndex + 1}</span>
                    <div className="lane-track">
                      {segments.map((segment) => (
                        <button
                          aria-label={`Seek camera ${cameraIndex + 1} track at ${segment.start} seconds`}
                          className="track-segment"
                          key={`${cameraIndex}-${segment.start}`}
                          onClick={() => seekTo(segment.start)}
                          style={{
                            left: `${(segment.start / 15) * 100}%`,
                            width: `${(segment.duration / 15) * 100}%`,
                            background: segment.color
                          }}
                          type="button"
                        />
                      ))}
                      <span className="playhead" style={{ left: `${(currentTime / Math.max(duration, 1)) * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </section>
          </section>

          <aside className="investigation-panel matches-panel">
            <div className="section-title">
              <FileSearch aria-hidden="true" size={18} />
              <h3>Candidate Matches</h3>
              <span>{selectedSubject.candidates.length}</span>
            </div>
            <div className="candidate-groups">
              <section>
                <h4>Ranked gallery evidence</h4>
                <div className="candidate-list">
                  {selectedSubject.candidates.map((candidate) => (
                    <button
                      className={`candidate-card ${candidate.same_identity ? "match-high" : "match-low"} ${candidate.rank === selectedCandidateRank ? "selected" : ""}`}
                      key={candidate.rank}
                      onClick={() => openCandidate(candidate.rank)}
                      type="button"
                    >
                      <span className="score-badge">{candidate.score.toFixed(3)}</span>
                      <img loading="lazy" src={candidate.image} alt={`Rank ${candidate.rank} candidate for ID ${selectedSubject.id}`} />
                      <div>
                        <strong>Rank {candidate.rank} · {selectedSubject.gallery_source}</strong>
                        <span>{candidate.same_identity ? "Same labeled identity" : "Distractor"}</span>
                        <small>Legacy replay score · click to inspect</small>
                      </div>
                    </button>
                  ))}
                </div>
              </section>
            </div>
            <p className="candidate-disclaimer">
              Candidate scores reproduce the archived application output. They rank visual evidence
              and are not proof of identity.
            </p>
          </aside>
        </section>
      )}
    </div>
  );
}
