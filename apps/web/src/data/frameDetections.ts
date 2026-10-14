import fixtureJson from "./frame-detections.json";

/**
 * Every detection the detector returned on each processed frame, built by
 * tools/build_detection_fixture.py from the same run as the person-search
 * index. The index keeps only a few views of a capped set of tracks, so a frame
 * redrawn from it misses most of the people in it; this is the full record.
 *
 * Rows are [x1, y1, x2, y2, score, trackId] in source pixels; trackId is null
 * when the box was never associated to an active track.
 */
export type FrameDetectionRow = [number, number, number, number, number, number | null];

type FrameDetectionFixture = {
  schema_version: number;
  generated_with: { model: string; frame_stride: number; store_floor: number };
  videos: { video_index: number; fps: number; frames: Record<string, FrameDetectionRow[]> }[];
};

export const frameDetections = fixtureJson as unknown as FrameDetectionFixture;
