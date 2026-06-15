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

const CardHeader = defineComponent({
  name: "CardHeader",
  description: "Displays the title and subtitle for a response card.",
  props: z.object({
    title: z.string(),
    subtitle: z.string().optional(),
  }),
  component: ({ props }) => (
    <header style={{ display: "grid", gap: 4 }}>
      <h2 style={{ margin: 0, fontSize: 18, lineHeight: 1.25, color: "#0f172a" }}>{props.title}</h2>
      {props.subtitle ? <p style={{ margin: 0, color: "#64748b", fontSize: 13 }}>{props.subtitle}</p> : null}
    </header>
  ),
});

const TextContent = defineComponent({
  name: "TextContent",
  description: "Displays short text content.",
  props: z.object({
    text: z.string(),
    variant: z.string().optional(),
  }),
  component: ({ props }) => (
    <p
      style={{
        margin: 0,
        color: props.variant === "small" ? "#64748b" : "#334155",
        fontSize: props.variant === "small" ? 12 : 14,
        lineHeight: 1.5,
      }}
    >
      {props.text}
    </p>
  ),
});

const ListItem = defineComponent({
  name: "ListItem",
  description: "Displays a label and supporting text inside a list.",
  props: z.object({
    title: z.string(),
    body: z.string().optional(),
  }),
  component: ({ props }) => (
    <li style={{ display: "grid", gap: 2 }}>
      <strong style={{ color: "#0f172a", fontSize: 13 }}>{props.title}</strong>
      {props.body ? <span style={{ color: "#64748b", fontSize: 12, lineHeight: 1.45 }}>{props.body}</span> : null}
    </li>
  ),
});

const ListBlock = defineComponent({
  name: "ListBlock",
  description: "Displays a compact list of response steps or metrics.",
  props: z.object({
    items: z.array(ListItem.ref),
    style: z.string().optional(),
  }),
  component: ({ props, renderNode }) => {
    const ordered = props.style === "number";
    const Tag = ordered ? "ol" : "ul";
    return (
      <Tag
        style={{
          margin: 0,
          paddingLeft: ordered ? 20 : 18,
          display: "grid",
          gap: 8,
        }}
      >
        {renderNode(props.items)}
      </Tag>
    );
  },
});

const Col = defineComponent({
  name: "Col",
  description: "Column data for a table.",
  props: z.object({
    label: z.string(),
    values: z.array(z.any()),
    type: z.string().optional(),
  }),
  component: () => null,
});

const Series = defineComponent({
  name: "Series",
  description: "Series data for a chart.",
  props: z.object({
    name: z.string(),
    values: z.array(z.number()),
  }),
  component: () => null,
});

const Table = defineComponent({
  name: "Table",
  description: "Renders column-oriented table data.",
  props: z.object({
    columns: z.array(Col.ref),
  }),
  component: ({ props }) => {
    const columns = props.columns || [];
    const rowCount = Math.max(...columns.map((column) => column.values?.length || 0), 0);
    return (
      <section style={cardStyle}>
        <div style={{ maxHeight: 320, overflow: "auto" }}>
          <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 13 }}>
            <thead>
              <tr>
                {columns.map((column, index) => (
                  <th key={`${column.label}-${index}`} style={{ borderBottom: "1px solid #e2e8f0", padding: 8, textAlign: "left" }}>
                    {column.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: rowCount }).map((_, rowIndex) => (
                <tr key={rowIndex}>
                  {columns.map((column, columnIndex) => (
                    <td key={`${rowIndex}-${columnIndex}`} style={{ borderBottom: "1px solid #e2e8f0", padding: 8, verticalAlign: "top" }}>
                      {String(column.values?.[rowIndex] ?? "")}
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
  description: "Renders a simple bar chart from row data or OpenUI standard series data.",
  props: z.object({
    first: z.any(),
    second: z.any(),
    third: z.any().optional(),
    fourth: z.any().optional(),
    fifth: z.any().optional(),
  }),
  component: ({ props }) => {
    const isSeriesChart = Array.isArray(props.first) && Array.isArray(props.second);
    const labels = isSeriesChart ? props.first.map(String) : [];
    const series = isSeriesChart ? props.second : [];
    const rows = isSeriesChart
      ? labels.map((label, index) => ({
          label,
          value: Number(series[0]?.values?.[index] ?? 0),
          series: series[0]?.name || "",
        }))
      : Array.isArray(props.fourth)
        ? props.fourth
        : [];
    const title = isSeriesChart ? String(props.fifth || props.fourth || "Chart") : String(props.first || "Chart");
    const xColumn = isSeriesChart ? "label" : String(props.second || "label");
    const yColumn = isSeriesChart ? "value" : String(props.third || "value");
    const values = rows.map((row) => Number(row[yColumn])).filter(Number.isFinite);
    const max = Math.max(...values, 1);
    return (
      <section style={cardStyle}>
        <h3 style={{ margin: "0 0 8px", fontSize: 15, color: "#0f172a" }}>{title}</h3>
        <div style={{ display: "grid", gap: 8 }}>
          {rows.map((row, index) => {
            const value = Number(row[yColumn]) || 0;
            const width = Math.max(3, (value / max) * 100);
            return (
              <div key={index} style={{ display: "grid", gridTemplateColumns: "minmax(70px, 140px) minmax(0, 1fr) minmax(44px, 72px)", gap: 10, alignItems: "center" }}>
                <div style={{ color: "#334155", fontSize: 12, overflowWrap: "anywhere" }}>{String(row[xColumn] ?? "")}</div>
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

const Callout = defineComponent({
  name: "Callout",
  description: "Displays a highlighted message.",
  props: z.object({
    tone: z.string(),
    title: z.string(),
    body: z.string().optional(),
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
          display: "grid",
          gap: 4,
        }}
      >
        <strong style={{ fontSize: 13 }}>{props.title}</strong>
        {props.body ? <span style={{ fontSize: 13, lineHeight: 1.45 }}>{props.body}</span> : null}
      </section>
    );
  },
});

const FollowUpItem = defineComponent({
  name: "FollowUpItem",
  description: "Displays a suggested follow-up prompt.",
  props: z.object({
    text: z.string(),
  }),
  component: ({ props }) => (
    <button
      type="button"
      style={{
        border: "1px solid #cbd5e1",
        borderRadius: 8,
        background: "#fff",
        color: "#0f172a",
        padding: "8px 10px",
        fontSize: 13,
        textAlign: "left",
        cursor: "pointer",
      }}
      onClick={() => {
        const textbox = document.querySelector("textarea, input[type='text']");
        if (textbox) {
          textbox.value = props.text;
          textbox.dispatchEvent(new Event("input", { bubbles: true }));
          textbox.focus();
        }
      }}
    >
      {props.text}
    </button>
  ),
});

const FollowUpBlock = defineComponent({
  name: "FollowUpBlock",
  description: "Displays suggested follow-up prompts.",
  props: z.object({
    items: z.array(FollowUpItem.ref),
  }),
  component: ({ props, renderNode }) => (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
      {renderNode(props.items)}
    </div>
  ),
});

const CodeBlock = defineComponent({
  name: "CodeBlock",
  description: "Displays code or structured text.",
  props: z.object({
    language: z.string(),
    code: z.string(),
  }),
  component: ({ props }) => (
    <pre
      style={{
        margin: 0,
        padding: 12,
        borderRadius: 8,
        background: "#0f172a",
        color: "#e2e8f0",
        overflow: "auto",
        fontSize: 12,
      }}
    >
      <code>{props.code}</code>
    </pre>
  ),
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

const Card = defineComponent({
  name: "Card",
  description: "Root card layout for OpenUI chat responses.",
  props: z.object({
    children: z.array(
      z.union([
        CardHeader.ref,
        TextContent.ref,
        ListBlock.ref,
        Table.ref,
        BarChart.ref,
        Callout.ref,
        FollowUpBlock.ref,
        CodeBlock.ref,
        InsightCard.ref,
        Notice.ref,
        MetricGrid.ref,
        DataTable.ref,
        Histogram.ref,
      ])
    ),
  }),
  component: ({ props, renderNode }) => (
    <article style={{ ...cardStyle, display: "grid", gap: 12 }}>
      {renderNode(props.children)}
    </article>
  ),
});

const Root = defineComponent({
  name: "Root",
  description: "Root layout for rendered analysis components.",
  props: z.object({
    children: z.array(z.union([Card.ref, InsightCard.ref, Notice.ref, MetricGrid.ref, DataTable.ref, BarChart.ref, Histogram.ref])),
  }),
  component: ({ props, renderNode }) => <div style={{ display: "grid", gap: 12 }}>{renderNode(props.children)}</div>,
});

const library = createLibrary({
  root: "Root",
  components: [
    Root,
    Card,
    CardHeader,
    TextContent,
    ListBlock,
    ListItem,
    Table,
    Col,
    Series,
    Callout,
    FollowUpBlock,
    FollowUpItem,
    CodeBlock,
    InsightCard,
    Notice,
    Metric,
    MetricGrid,
    DataTable,
    BarChart,
    Histogram,
  ],
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
