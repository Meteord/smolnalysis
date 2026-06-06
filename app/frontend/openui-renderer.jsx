import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Renderer, createLibrary, defineComponent } from "@openuidev/react-lang";
import { z } from "zod/v4";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: 8,
  padding: 14,
  background: "#fff",
  boxShadow: "0 6px 20px rgba(15, 23, 42, 0.04)",
};

const InsightCard = defineComponent({
  name: "InsightCard",
  description: "Displays a title and short analysis text.",
  props: z.object({
    title: z.string(),
    body: z.string(),
  }),
  component: ({ props }) => (
    <section style={cardStyle}>
      <h3 style={{ margin: "0 0 8px", fontSize: 15, color: "#0f172a" }}>{props.title}</h3>
      <p style={{ margin: 0, color: "#475569", fontSize: 14 }}>{props.body}</p>
    </section>
  ),
});

const Notice = defineComponent({
  name: "Notice",
  description: "Shows an informational or warning message.",
  props: z.object({
    message: z.string(),
    tone: z.string().optional(),
  }),
  component: ({ props }) => {
    const warning = props.tone === "warning";
    return (
      <section
        style={{
          borderRadius: 8,
          padding: "12px 14px",
          border: `1px solid ${warning ? "#fde68a" : "#bae6fd"}`,
          background: warning ? "#fffbeb" : "#f0f9ff",
          color: warning ? "#78350f" : "#0c4a6e",
        }}
      >
        {props.message}
      </section>
    );
  },
});

const Metric = defineComponent({
  name: "Metric",
  description: "Displays a compact metric value.",
  props: z.object({
    label: z.string(),
    value: z.string(),
    caption: z.string().optional(),
  }),
  component: ({ props }) => (
    <div
      style={{
        border: "1px solid #e2e8f0",
        borderRadius: 8,
        padding: 12,
        background: "#f8fafc",
      }}
    >
      <span style={{ display: "block", color: "#64748b", fontSize: 12 }}>{props.label}</span>
      <strong style={{ display: "block", marginTop: 4, fontSize: 21, color: "#0f172a" }}>{props.value}</strong>
      {props.caption ? <small style={{ display: "block", marginTop: 4, color: "#64748b" }}>{props.caption}</small> : null}
    </div>
  ),
});

const MetricGrid = defineComponent({
  name: "MetricGrid",
  description: "Renders metric cards in a responsive grid.",
  props: z.object({
    metrics: z.array(Metric.ref),
  }),
  component: ({ props, renderNode }) => (
    <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 10 }}>
      {renderNode(props.metrics)}
    </section>
  ),
});

const DataTable = defineComponent({
  name: "DataTable",
  description: "Renders rows of tabular data.",
  props: z.object({
    title: z.string(),
    rows: z.array(z.record(z.string(), z.any())),
  }),
  component: ({ props }) => {
    const columns = Object.keys(props.rows?.[0] || {});
    return (
      <section style={cardStyle}>
        <h3 style={{ margin: "0 0 8px", fontSize: 15, color: "#0f172a" }}>{props.title}</h3>
        <div style={{ maxHeight: 320, overflow: "auto" }}>
          <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 13 }}>
            <thead>
              <tr>
                {columns.map((column) => (
                  <th key={column} style={{ borderBottom: "1px solid #e2e8f0", padding: 8, textAlign: "left" }}>
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {props.rows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {columns.map((column) => (
                    <td key={column} style={{ borderBottom: "1px solid #e2e8f0", padding: 8, verticalAlign: "top" }}>
                      {String(row[column] ?? "")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    );
  },
});

const BarChart = defineComponent({
  name: "BarChart",
  description: "Renders a simple horizontal bar chart from row data.",
  props: z.object({
    title: z.string(),
    xColumn: z.string(),
    yColumn: z.string(),
    rows: z.array(z.record(z.string(), z.any())),
  }),
  component: ({ props }) => {
    const values = props.rows.map((row) => Number(row[props.yColumn])).filter(Number.isFinite);
    const max = Math.max(...values, 1);
    return (
      <section style={cardStyle}>
        <h3 style={{ margin: "0 0 8px", fontSize: 15, color: "#0f172a" }}>{props.title}</h3>
        <div style={{ display: "grid", gap: 8 }}>
          {props.rows.map((row, index) => {
            const value = Number(row[props.yColumn]) || 0;
            const width = Math.max(3, (value / max) * 100);
            return (
              <div key={index} style={{ display: "grid", gridTemplateColumns: "minmax(70px, 140px) minmax(0, 1fr) minmax(44px, 72px)", gap: 10, alignItems: "center" }}>
                <div style={{ color: "#334155", fontSize: 12, overflowWrap: "anywhere" }}>{String(row[props.xColumn] ?? "")}</div>
                <div style={{ height: 14, borderRadius: 999, background: "#e2e8f0", overflow: "hidden" }}>
                  <div style={{ height: "100%", width: `${width}%`, borderRadius: 999, background: "linear-gradient(90deg, #2563eb, #0891b2)" }} />
                </div>
                <div style={{ color: "#334155", fontSize: 12 }}>{value.toLocaleString()}</div>
              </div>
            );
          })}
        </div>
      </section>
    );
  },
});

const Histogram = defineComponent({
  name: "Histogram",
  description: "Renders a simple histogram from numeric values.",
  props: z.object({
    title: z.string(),
    column: z.string(),
    values: z.array(z.number()),
  }),
  component: ({ props }) => {
    const values = props.values.filter(Number.isFinite);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    const counts = Array.from({ length: 12 }, () => 0);
    values.forEach((value) => {
      const index = Math.min(11, Math.floor(((value - min) / span) * 12));
      counts[index] += 1;
    });
    const top = Math.max(...counts, 1);
    return (
      <section style={cardStyle}>
        <h3 style={{ margin: "0 0 8px", fontSize: 15, color: "#0f172a" }}>{props.title}</h3>
        <p style={{ margin: 0, color: "#475569", fontSize: 14 }}>{props.column}</p>
        <div style={{ display: "flex", gap: 4, alignItems: "end", height: 180, paddingTop: 8 }}>
          {counts.map((count, index) => (
            <div
              key={index}
              title={String(count)}
              style={{
                flex: 1,
                minWidth: 8,
                height: `${Math.max(4, (count / top) * 100)}%`,
                borderRadius: "4px 4px 0 0",
                background: "linear-gradient(180deg, #0ea5e9, #2563eb)",
              }}
            />
          ))}
        </div>
      </section>
    );
  },
});

const Root = defineComponent({
  name: "Root",
  description: "Root layout for rendered analysis components.",
  props: z.object({
    children: z.array(z.union([InsightCard.ref, Notice.ref, MetricGrid.ref, DataTable.ref, BarChart.ref, Histogram.ref])),
  }),
  component: ({ props, renderNode }) => <div style={{ display: "grid", gap: 12 }}>{renderNode(props.children)}</div>,
});

const library = createLibrary({
  root: "Root",
  components: [Root, InsightCard, Notice, Metric, MetricGrid, DataTable, BarChart, Histogram],
});

function decodeResponse(encoded) {
  if (!encoded) return "";
  const bytes = Uint8Array.from(atob(encoded), (char) => char.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

function OpenUIApp({ response }) {
  const [errors, setErrors] = useState([]);

  useEffect(() => {
    setErrors([]);
  }, [response]);

  if (errors.length > 0) {
    return (
      <section
        style={{
          borderRadius: 8,
          padding: "12px 14px",
          border: "1px solid #fde68a",
          background: "#fffbeb",
          color: "#78350f",
        }}
      >
        OpenUI could not render this response: {errors.map((error) => error.message).join("; ")}
      </section>
    );
  }

  return (
    <Renderer
      library={library}
      response={response}
      isStreaming={false}
      onError={(nextErrors) => setErrors(nextErrors || [])}
    />
  );
}

window.SmolnalysisOpenUIRenderer = {
  mountPoint(mountPoint) {
    if (!mountPoint) return;
    const encoded = mountPoint.dataset.openuiEncoded || "";
    if (mountPoint.__lastOpenUIEncoded === encoded) return;
    mountPoint.__lastOpenUIEncoded = encoded;
    const response = decodeResponse(encoded || "");
    if (!mountPoint.__smolnalysisRoot) {
      mountPoint.__smolnalysisRoot = createRoot(mountPoint);
    }
    mountPoint.__smolnalysisRoot.render(<OpenUIApp response={response} />);
  },
  mount(element, encoded) {
    const mountPoint = element.querySelector("[data-openui-mount]");
    if (!mountPoint) return;
    mountPoint.dataset.openuiEncoded = encoded || "";
    this.mountPoint(mountPoint);
  },
  mountAll() {
    document.querySelectorAll("[data-openui-mount]").forEach((mountPoint) => this.mountPoint(mountPoint));
  },
};

document.documentElement.dataset.smolnalysisOpenuiLoaded = "true";
window.SmolnalysisOpenUIRenderer.mountAll();
new MutationObserver(() => window.SmolnalysisOpenUIRenderer.mountAll()).observe(document.body, {
  attributes: true,
  childList: true,
  subtree: true,
});
setInterval(() => window.SmolnalysisOpenUIRenderer.mountAll(), 500);
