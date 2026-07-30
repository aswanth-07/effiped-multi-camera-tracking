export type Camera = {
  id: string;
  label: string;
  position: "top-left" | "top-right" | "bottom-left" | "bottom-right";
  time: string;
};

export type Appearance = {
  camera: string;
  similarity: number;
  bbox: [number, number, number, number];
};

export type Subject = {
  id: string;
  label: string;
  color: string;
  appearances: Appearance[];
};

export type ResultFixture = {
  project: {
    brand: string;
    title: string;
    author: string;
    guide: string;
    award: string;
    event: string;
    track: string;
    institution: string;
  };
  verified_contest_system: {
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
  contest_submission_snapshot: {
    reported_parameters_m: number;
    reported_fps: number;
    reported_rank1_cross: number;
    composite_gain_pp: number;
    note: string;
  };
  post_contest_evolution: {
    partjde: Record<string, string | number>;
    boxjde: Record<string, string | number>;
  };
  demo_case: {
    id: string;
    title: string;
    description: string;
    source_media: string;
    cameras: Camera[];
    subjects: Subject[];
    session_diagnostic: Record<string, string | number>;
  };
};

