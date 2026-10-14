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
    model: string;
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
};

