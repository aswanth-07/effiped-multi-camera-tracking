import { Callout, Code, DataTable, PageHead, Section } from "../components/page";
import { results } from "../data/results";
import { repository } from "../lib/site";

export function Deploy() {
  const footprint = results.system_benchmarks.footprint;

  return (
    <>
      <PageHead
        index="04"
        label="Run it"
        lede="The hosted workbench needs nothing. Running real inference needs a checkout, an authorized checkpoint and a GPU, and this page is honest about what happens when you do not have the checkpoint."
        title="Run it locally"
      />

      <Section label="Install" title="From a clean clone to a running service.">
        <Code label="Install and start the Python service">{`python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux and macOS: source .venv/bin/activate

pip install -e ".[runtime]"
effiped-app                      # http://127.0.0.1:8000`}</Code>
        <p className="prose">
          The <code>runtime</code> extra adds FastAPI, uvicorn, OpenCV and the multipart parser on top
          of the base PyTorch stack. Without it the service will not import. Install editable rather
          than plain: two of the console commands resolve paths relative to the repository root, and
          from a site-packages install those point outside your checkout.
        </p>
        <Code label="Install and start the web interface">{`npm install --prefix apps/web
npm run dev --prefix apps/web    # http://127.0.0.1:5173`}</Code>
        <p className="prose">
          The interface runs on its own with no Python at all, which is what the hosted build is. When
          the service is also running it proxies to it, and the same panels drive real inference.
        </p>
      </Section>

      <Section label="Expected state" title="A healthy install reports every model unavailable.">
        <Callout title="This is not a broken install" tone="warning">
          <p>
            <code>pip install</code> succeeds, <code>effiped-app</code> starts,{" "}
            <code>/api/health</code> returns ok, and every model reports{" "}
            <code>available: false</code>. No checkpoint is published, because redistribution terms
            across P-DESTRE, MOT17 and MOT20, SOMPT22 and CrowdHuman are unresolved for trained
            parameters. A single unresolved source is enough to hold publication.
          </p>
          <p>
            Place an authorized checkpoint in <code>EFFIPED_WEIGHTS_DIR</code> and the same interface
            runs real CUDA inference. The service never reveals the path it probed, which is asserted
            by a test.
          </p>
        </Callout>
      </Section>

      <Section label="Configuration" title="Environment only, with defaults outside the repository.">
        <DataTable
          label="Environment variables"
          caption="Weights and runtime state default under the user cache directory, so nothing lands in version control by accident."
          columns={[
            { key: "name", label: "Variable" },
            { key: "default", label: "Default" },
            { key: "does", label: "What it controls" }
          ]}
          rows={[
            {
              key: "w",
              cells: {
                name: <code>EFFIPED_WEIGHTS_DIR</code>,
                default: <code>~/.cache/effiped/weights</code>,
                does: "Where an authorized checkpoint is read from"
              }
            },
            {
              key: "r",
              cells: {
                name: <code>EFFIPED_RUNTIME_DIR</code>,
                default: <code>~/.cache/effiped/runtime</code>,
                does: "Job state, extracted crops and indices"
              }
            },
            {
              key: "d",
              cells: {
                name: <code>EFFIPED_DEVICE</code>,
                default: <code>auto</code>,
                does: "Falls back to CPU. An explicit cuda request with no CUDA available fails immediately with a clear message"
              }
            },
            {
              key: "u",
              cells: {
                name: <code>EFFIPED_MAX_UPLOAD_MB</code>,
                default: <code>512</code>,
                does: "Per-video upload ceiling"
              }
            },
            {
              key: "o",
              cells: {
                name: <code>EFFIPED_ALLOWED_ORIGINS</code>,
                default: <code>127.0.0.1:5173</code>,
                does: "CORS allowlist, comma separated"
              }
            }
          ]}
          source={{ label: "src/effiped/settings.py", href: `${repository}/blob/main/src/effiped/settings.py` }}
        />
      </Section>

      <Section label="Operating notes" title="Things worth knowing before you point it at real footage.">
        <ul className="limits">
          <li>
            <strong>The service binds loopback and has no authentication.</strong> That is safe only
            because of the bind. The container image binds all interfaces, so publishing that port is
            a decision requiring its own review.
          </li>
          <li>
            <strong>Job state is not reclaimed on a timer.</strong> Deleting a job is the only path
            that removes its uploaded video, crops and indices, and it refuses while the job is still
            running.
          </li>
          <li>
            <strong>Hardware.</strong> An NVIDIA GPU is recommended. The published throughput of about{" "}
            {footprint.pipeline_fps_approx} FPS is one device at {footprint.input_resolution} and is
            approximate.
          </li>
          <li>
            <strong>Treat the inputs as sensitive.</strong> Video, crops, descriptors and candidate
            rankings are personal data. The security policy in the repository says what to do with
            them.
          </li>
        </ul>
      </Section>

      <Section label="Licensing" title="Two licences, deliberately.">
        <p className="prose">
          The software is Apache-2.0. P-DESTRE-derived media is separately licensed CC BY-NC-SA 4.0,
          and every derived asset is hashed and attributed in a manifest that a validator re-checks on
          every CI run. No dataset, source video, annotation archive, checkpoint or person-level
          benchmark record is committed. The model is not distributed under Apache-2.0, because it is
          not distributed at all.
        </p>
      </Section>
    </>
  );
}
