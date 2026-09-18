import { LiveConsole } from "../components/LiveConsole";
import { Workbench } from "../components/Workbench";
import { PageHead } from "../components/page";

export function WorkbenchPage() {
  const mode = import.meta.env.VITE_APP_MODE ?? "demo";

  return (
    <>
      <PageHead
        index="01"
        label={mode === "demo" ? "Precomputed replay" : "Local inference"}
        lede={
          mode === "demo"
            ? "Four P-DESTRE clips from session 12-11-2019_3 are already attached, as though you had uploaded them. Every control is live, and pressing Run replays the archived result for those settings rather than inferring in your browser."
            : "Connected to the local service. Uploaded clips are processed on your own GPU and removed when you delete the job."
        }
        title="Identity review workbench"
      />
      {mode === "live" ? <LiveConsole /> : <Workbench />}
    </>
  );
}
