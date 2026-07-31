export type DemoVideo = {
  id: string;
  label: string;
  description: string;
  source: string;
  poster: string;
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
