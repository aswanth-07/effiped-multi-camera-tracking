export type DemoVideo = {
  id: string;
  label: string;
  description: string;
  source: string;
  poster: string;
};

/* ---- Precomputed person-search fixture (apps/web/src/data/person-search.json) ---- */

export type WorkbenchVideo = {
  id: number;
  name: string;
  label: string;
  file_name: string;
  duration_s: number;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
  status: string;
  source: string;
  tracked: string;
  poster: string;
};

export type WorkbenchPerson = {
  id: string;
  video_index: number;
  track_id: number;
  first_frame: number;
  last_frame: number;
  first_time_s: number;
  last_time_s: number;
  num_samples: number;
  best_score: number;
  caption: string;
  crop: string | null;
};

export type WorkbenchAppearance = {
  id: string;
  video_index: number;
  track_id: number;
  frame_index: number;
  frame_idx: number;
  time_s: number;
  score: number;
  caption: string;
  crop: string | null;
  /** [x1, y1, x2, y2] in the source clip's own pixel space. */
  bbox: [number, number, number, number];
};

export type WorkbenchMatch = {
  similarity: number;
  same_video: boolean;
  person: WorkbenchPerson;
};

export type PersonSearchFixture = {
  schema_version: number;
  generated_with: {
    checkpoint: string;
    checkpoint_key: string;
    note: string;
    settings: Record<string, number | string>;
  };
  job: {
    job_id: string;
    status: string;
    message: string;
    progress: number;
    processed_frame_sets: number;
    total_frame_sets: number;
    people_count: number;
  };
  videos: WorkbenchVideo[];
  people: WorkbenchPerson[];
  details: Record<string, { appearances: WorkbenchAppearance[] }>;
  matches: Record<string, WorkbenchMatch[]>;
  detection: { input: string; output: string };
};

export type DemoCandidate = {
  rank: number;
  image: string;
  score: number;
  same_identity: boolean;
};

export type DemoSubject = {
  id: string;
  label: string;
  query_source: string;
  gallery_source: string;
  query_image: string;
  candidates: DemoCandidate[];
};

export type ResultFixture = {
  schema_version: number;
  project: {
    brand: string;
    application: string;
    title: string;
    author: string;
    summary: string;
  };
  system_benchmarks: {
    model: string;
    pdestre: {
      validation: { protocol: string; rank1_cross: number; detection_map50: number };
      test: { protocol: string; rank1_cross: number; detection_map50: number };
    };
    mot17: { protocol: string; mota: number; idf1: number; hota: number };
    footprint: {
      parameters_m: number;
      pipeline_fps_approx: number;
      device: string;
      input_resolution: string;
      descriptor_dim: number;
    };
  };
  research_extensions: {
    partjde: Record<string, string | number>;
    boxjde: Record<string, string | number>;
  };
  demo_case: {
    id: string;
    title: string;
    description: string;
    videos: DemoVideo[];
    subjects: DemoSubject[];
    session_diagnostic: {
      cameras: number;
      frames: number;
      local_tracks: number;
      cross_camera_ids: number;
      pairwise_association_precision: number;
      playback_fps: number;
      label: string;
    };
  };
};
