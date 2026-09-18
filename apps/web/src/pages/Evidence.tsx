import { Callout, DataTable, Label, PageHead, Section } from "../components/page";
import { results } from "../data/results";
import { sessionStats } from "../data/personSearch";
import { boxjdeRepository, repository } from "../lib/site";

const fixtureHref = `${repository}/blob/main/research/results/summary.json`;

export function Evidence() {
  const benchmark = results.system_benchmarks;
  const partjde = results.research_extensions.partjde;
  const boxjde = results.research_extensions.boxjde;

  return (
    <>
      <PageHead
        index="03"
        label="Evidence"
        lede="Every figure this project publishes lives in one checked-in fixture, and a validator in CI pins each value and the claim language around it. Changing a published number requires changing the validator too, which is the intended friction."
        title="Evidence and protocols"
      >
        <p className="page-head__source">
          <Label tone="link">
            <a href={fixtureHref} rel="noreferrer" target="_blank">
              research/results/summary.json
            </a>
          </Label>
        </p>
      </PageHead>

      <Section
        label="Delivered system"
        title="Standard benchmarks."
        lede="The integrated system, measured under three protocols that answer different questions."
      >
        <DataTable
          label="Standard benchmark results by protocol"
          caption="Validation sits about 1.5 points above test on cross-camera Rank-1, which is the ordinary direction and a small enough gap to report both rather than only the better one."
          columns={[
            { key: "eval", label: "Evaluation" },
            { key: "metric", label: "Metric" },
            { key: "value", label: "Value", numeric: true }
          ]}
          rows={[
            {
              key: "v1",
              cells: {
                eval: "P-DESTRE fold-0 validation",
                metric: "Cross-camera Rank-1",
                value: benchmark.pdestre.validation.rank1_cross.toFixed(2)
              },
              emphasis: "positive"
            },
            {
              key: "v2",
              cells: {
                eval: "P-DESTRE fold-0 validation",
                metric: "Detection mAP@0.5",
                value: benchmark.pdestre.validation.detection_map50.toFixed(2)
              }
            },
            {
              key: "t1",
              cells: {
                eval: "P-DESTRE fold-0 test",
                metric: "Cross-camera Rank-1",
                value: benchmark.pdestre.test.rank1_cross.toFixed(2)
              }
            },
            {
              key: "t2",
              cells: {
                eval: "P-DESTRE fold-0 test",
                metric: "Detection mAP@0.5",
                value: benchmark.pdestre.test.detection_map50.toFixed(2)
              }
            },
            { key: "m1", cells: { eval: "MOT17 val-half", metric: "MOTA", value: benchmark.mot17.mota.toFixed(2) } },
            { key: "m2", cells: { eval: "MOT17 val-half", metric: "IDF1", value: benchmark.mot17.idf1.toFixed(2) } },
            { key: "m3", cells: { eval: "MOT17 val-half", metric: "HOTA", value: benchmark.mot17.hota.toFixed(2) } }
          ]}
          source={{ label: "summary.json", href: fixtureHref }}
        />
        <p className="prose">
          The P-DESTRE rows use Protocol D for validation and Protocol E for test. The MOT17 rows use
          Protocol A at minimum visibility 0.3 with BoT-SORT, a 0.30 detection threshold and 0.60 NMS,
          on sequences held out of the training list. Tier-1 is {benchmark.footprint.parameters_m}M
          parameters at roughly {benchmark.footprint.pipeline_fps_approx} FPS end to end on an{" "}
          {benchmark.footprint.device} at {benchmark.footprint.input_resolution}.
        </p>
      </Section>

      <Section
        label="Descriptor research"
        title="A different boundary, and not comparable to the rows above."
        lede="Two follow-on studies asked one question: how much does the way a descriptor is read out of the feature map matter, holding the rest roughly fixed."
      >
        <DataTable
          label="Descriptor readout studies"
          caption="Both studies vary readout under matched conditions. BoxJDE reports a smaller model with a larger gain, which is the sharpest available form of the claim."
          columns={[
            { key: "study", label: "Study" },
            { key: "boundary", label: "Boundary" },
            { key: "gain", label: "Rank-1 gain", numeric: true },
            { key: "params", label: "Params", numeric: true }
          ]}
          rows={[
            {
              key: "part",
              cells: {
                study: "PartJDE, four-strip readout",
                boundary: "Matched fold-0 validation",
                gain: `+${partjde.matched_part_readout_gain_pp} pp`,
                params: `${partjde.parameters_m}M`
              }
            },
            {
              key: "box",
              cells: {
                study: "BoxJDE, full-person readout",
                boundary: "Constructed five-fold per-date ablation",
                gain: `+${boxjde.source_detected_rank1_gain_pp} pp`,
                params: `${boxjde.parameters_m}M`
              },
              emphasis: "positive"
            }
          ]}
          source={{ label: "BoxJDE repository", href: boxjdeRepository }}
        />
        <Callout title="BoxJDE is a constructed ablation" tone="warning">
          <p>
            Its five-fold per-date split was constructed for the readout comparison and is
            deliberately not the official P-DESTRE Task 4 protocol. Quoting it as a Task 4 result
            would be a claim error. Its primary evidence and technical report live in a separate
            repository, so this page reports the summary figures rather than the study itself.
          </p>
        </Callout>
      </Section>

      <Section
        label="Demo session"
        title="The archived session behind the workbench."
        lede="These numbers describe one recorded session of the application. They are a diagnostic, not a benchmark, and the fixture labels them that way."
      >
        <DataTable
          label="Per camera-pair association precision for the demo session"
          caption={`${results.demo_case.session_diagnostic.label}. Precision here is the share of proposed cross-camera associations that were correct in this one session.`}
          columns={[
            { key: "pair", label: "Camera pair" },
            { key: "correct", label: "Correct", numeric: true },
            { key: "wrong", label: "Wrong", numeric: true },
            { key: "shared", label: "Shared", numeric: true },
            { key: "prec", label: "Precision", numeric: true }
          ]}
          rows={sessionStats.perPair.map((pair) => ({
            key: pair.pair,
            cells: {
              pair: pair.pair,
              correct: pair.correct,
              wrong: pair.wrong,
              shared: pair.shared,
              prec: pair.precision.toFixed(2)
            },
            emphasis: pair.precision >= 0.99 ? ("positive" as const) : undefined
          }))}
        />
        <p className="prose">
          The spread is the interesting part. Cameras 1 and 2 associate perfectly across 10 proposals,
          while cameras 3 and 4 fall to {sessionStats.perPair[5].precision.toFixed(2)}. Reporting the
          overall {sessionStats.precision.toFixed(2)} alone would hide that, and the pair that
          disagrees is the one worth looking at.
        </p>
      </Section>

      <Section label="Limits" title="What the evidence does not support.">
        <ul className="limits">
          <li>
            <strong>That a ranked match establishes identity.</strong> It is reviewable appearance
            evidence. Scores are uncalibrated: in the demo data, non-matches land between 0.997 and
            0.998 against matches at 0.998 to 0.999.
          </li>
          <li>
            <strong>That any figure transfers.</strong> Nothing here establishes behaviour for another
            site, population, camera network or operating condition.
          </li>
          <li>
            <strong>Any statement of significance.</strong> Every published value is a single
            measurement on one fold. No variance, interval, seed policy or repetition is reported.
          </li>
          <li>
            <strong>That the browser replay measures anything.</strong> It replays archived output and
            performs no inference.
          </li>
          <li>
            <strong>Independent reproducibility.</strong> The checkpoints that produced these numbers
            are not published, and no script in the repository re-derives them. The figures are
            attested by the fixture and defended against drift by a validator. They are not currently
            re-derivable from the repository alone.
          </li>
          <li>
            <strong>Fairness across subgroups.</strong> No such analysis exists, on a system that ranks
            people by appearance.
          </li>
        </ul>
      </Section>
    </>
  );
}
