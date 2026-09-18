import { Callout, DataTable, PageHead, Section, Step } from "../components/page";
import { Architecture } from "../components/Architecture";
import { results } from "../data/results";
import { repository } from "../lib/site";

export function System() {
  const footprint = results.system_benchmarks.footprint;

  return (
    <>
      <PageHead
        index="02"
        label="System"
        lede="One ConvNeXt V2 feature hierarchy serves detection and identity at the same stride. That is the whole design bet: a two-stage detector followed by a separate re-identification model does not fit the footprint this system needs to run on one GPU."
        title="How the system works"
      />

      <Section label="Data path" title="From camera frame to ranked candidate.">
        <Architecture />
        <div className="steps steps--dense">
          <Step n="01" title="Preprocess">
            Each frame is letterboxed to {footprint.input_resolution} and normalized with ImageNet
            statistics. The letterbox ratio and padding are kept so boxes can be mapped back to the
            original frame exactly.
          </Step>
          <Step n="02" title="Backbone and neck">
            ConvNeXt V2 Tiny produces P2 to P4. The shipped configuration uses a depth-3 P3 refiner
            rather than BiFPN, which is the lighter of the two available paths.
          </Step>
          <Step n="03" title="Stride-4 fusion">
            P2 is projected by a 3 by 3 convolution, because P2's value is its spatial detail, and
            combined with the refined P3 through BiFPN-style fast-normalized learned weights.
          </Step>
          <Step n="04" title="Detection head">
            CenterNet outputs a heatmap, width and height, an offset and an IoU quality branch.
            Decoding applies top-k, a confidence floor, soft-NMS and minimum box area and height.
          </Step>
          <Step n="05" title="Identity head">
            RoIAlign crops each detection to 32 by 8. Four vertical strips are fused by Coordinate
            Attention into a {footprint.descriptor_dim}-D descriptor, with BNNeck applied at
            evaluation time.
          </Step>
          <Step n="06" title="Association">
            BoT-SORT maintains camera-local tracks. Cross-view candidates are ranked by cosine
            similarity against descriptors from the other cameras.
          </Step>
        </div>
      </Section>

      <Section
        label="Design decisions"
        title="Four choices that shaped everything else."
      >
        <DataTable
          label="Design decisions and the constraints that forced them"
          caption="Each row is a decision the configuration records, and the constraint that forced it."
          columns={[
            { key: "choice", label: "Decision" },
            { key: "why", label: "Why" }
          ]}
          rows={[
            {
              key: "joint",
              cells: {
                choice: "One network for detection and identity",
                why: `${footprint.parameters_m}M parameters and about ${footprint.pipeline_fps_approx} FPS for the complete pipeline. A detector plus a separate re-identification model does not fit that budget, and the budget is what makes local single-GPU operation possible.`
              }
            },
            {
              key: "stride",
              cells: {
                choice: "Detection and identity both at stride 4",
                why: "Aerial and elevated footage is full of small, distant people. Coarser strides drop them before the identity head ever sees them."
              }
            },
            {
              key: "parts",
              cells: {
                choice: "Four body strips rather than one pooled vector",
                why: "A partially occluded person still contributes the strips that are visible. A diversity term during training prevents the strips collapsing onto the same feature, which is the failure a part-based readout invites."
              }
            },
            {
              key: "split",
              cells: {
                choice: "Local and cross-camera association kept separate",
                why: "Camera-local association is temporal and motion-aware. Cross-camera association has only appearance and must not assume continuity. Merging them would smuggle a continuity assumption into the cross-camera path."
              }
            }
          ]}
        />
      </Section>

      <Section label="Training" title="The schedule is a memory story.">
        <p className="prose">
          A micro-batch of 2 with 8 accumulation steps is an effective batch of 16, assembled two at
          a time at {footprint.input_resolution}. Mixed precision is off, which gives up the obvious
          throughput win rather than risk instability at that micro-batch. Read the whole training
          block as fitting a joint model, full-resolution activations and an 8,192-entry cross-batch
          memory onto one laptop GPU.
        </p>
        <p className="prose">
          The augmentation is shaped like surveillance rather than like web imagery: JPEG compression
          between quality 30 and 70, resolution degradation to between a quarter and a half, and
          gamma shift, layered on top of mosaic, cutout, blur, noise, colour jitter and random
          erasing. The distribution being modelled is compressed, low-resolution, variably lit camera
          footage.
        </p>
        <Callout title="Identity supervision" tone="note">
          <p>
            ArcFace over BNNeck carries the identity loss, with a triplet term backed by an
            8,192-entry cross-batch memory. That memory is what makes triplet mining viable at an
            effective batch of 16. Half of each identity batch is drawn across cameras, which targets
            the cross-camera metric directly at training time.
          </p>
        </Callout>
      </Section>

      <Section label="Interfaces" title="What the service exposes.">
        <DataTable
          label="Service routes"
          caption="The local service is loopback-only and has no authentication, which is safe only because of that bind. Full route list in the repository."
          columns={[
            { key: "route", label: "Route" },
            { key: "does", label: "What it does" }
          ]}
          rows={[
            { key: "h", cells: { route: <code>GET /api/health</code>, does: "Version and resolved device" } },
            { key: "m", cells: { route: <code>GET /api/models</code>, does: "Model catalogue and whether each checkpoint is present" } },
            { key: "j", cells: { route: <code>POST /api/person-search/jobs</code>, does: "Accept clips and return a job id immediately" } },
            { key: "s", cells: { route: <code>WS .../jobs/{"{id}"}/stream</code>, does: "Replay a monotonic progress event sequence" } },
            { key: "p", cells: { route: <code>GET .../people/{"{id}"}/matches</code>, does: "Ranked cross-video candidates for one person" } },
            { key: "d", cells: { route: <code>DELETE .../jobs/{"{id}"}</code>, does: "Remove the uploaded video and every generated asset" } }
          ]}
          source={{ label: "apps/api/main.py", href: `${repository}/blob/main/apps/api/main.py` }}
        />
      </Section>
    </>
  );
}
