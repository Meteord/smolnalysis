import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { openAIAdapter } from "@openuidev/react-headless";
import { FullScreen, openuiChatLibrary } from "@openuidev/react-ui";
import "../../node_modules/@openuidev/react-ui/dist/styles/index.css";

import "./openui-chat.css";

const CKAN_STORAGE_KEY = "smolnalysis.ckanEndpoint";
const CHAT_ADAPTER = "auto";
const RUNTIME_ROLE_KEYS = {
  general_agent: "general_agent",
  ckan_tool: "ckan_retrieval",
  data_analysis: "data_analysis",
  openui_translator: "openui_translator",
};

function modelArtifactLabel(repoId, filename, fallback = "Model") {
  if (repoId && filename) return `${repoId}/${filename}`;
  if (repoId) return repoId;
  if (filename) return filename;
  return fallback;
}

function runtimeForRole(runtimeStatus, roleKey) {
  const runtimeRoleKey = RUNTIME_ROLE_KEYS[roleKey] || roleKey;
  if (runtimeStatus?.roles && !Array.isArray(runtimeStatus.roles)) {
    return runtimeStatus.roles[runtimeRoleKey] || null;
  }
  if (Array.isArray(runtimeStatus?.roles) && runtimeStatus.roles.includes(runtimeRoleKey)) {
    return {
      model_repo_id: runtimeStatus.model,
      model_hub_url: runtimeStatus.model_hub_url,
      configured: runtimeStatus.configured,
    };
  }
  return null;
}

function CkanEndpointPanel({ onConnectionChange }) {
  const [defaultEndpoint, setDefaultEndpoint] = useState("https://opendata.muenchen.de/");
  const [endpoint, setEndpoint] = useState("https://opendata.muenchen.de/");
  const [status, setStatus] = useState({ ok: false, message: "Not connected." });
  const [isConnecting, setIsConnecting] = useState(false);

  useEffect(() => {
    let active = true;

    fetch("/api/ckan/default")
      .then((response) => response.json())
      .then((data) => {
        if (!active) return;
        const nextDefault = data.default_endpoint || data.base_url || "https://opendata.muenchen.de/";
        const saved = window.localStorage.getItem(CKAN_STORAGE_KEY);
        setDefaultEndpoint(nextDefault);
        setEndpoint(saved || nextDefault);
        setStatus(saved ? { ok: false, message: "Saved endpoint ready to connect." } : data);
      })
      .catch(() => {
        const saved = window.localStorage.getItem(CKAN_STORAGE_KEY);
        if (!active || !saved) return;
        setEndpoint(saved);
        setStatus({ ok: false, message: "Saved endpoint ready to connect." });
      });

    return () => {
      active = false;
    };
  }, []);

  const connect = async () => {
    setIsConnecting(true);
    setStatus({ ok: false, message: "Checking CKAN endpoint..." });
    try {
      const response = await fetch("/api/ckan/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ base_url: endpoint }),
      });
      const data = await response.json();
      setStatus(data);
      if (data.ok && data.base_url) {
        setEndpoint(data.base_url);
        window.localStorage.setItem(CKAN_STORAGE_KEY, data.base_url);
        onConnectionChange?.({ connected: true, base_url: data.base_url });
      }
    } catch {
      setStatus({ ok: false, message: "Could not contact the local CKAN connector." });
    } finally {
      setIsConnecting(false);
    }
  };

  const reset = () => {
    window.localStorage.removeItem(CKAN_STORAGE_KEY);
    setEndpoint(defaultEndpoint);
    setStatus({ ok: false, message: "Reset to the default CKAN endpoint." });
    onConnectionChange?.({ connected: false, base_url: defaultEndpoint });
  };

  return (
    <section className="ckan-panel" aria-label="CKAN endpoint configuration">
      <div className="ckan-panel__copy">
        <span className="ckan-panel__label">CKAN endpoint</span>
        <span className={`ckan-panel__status ${status.ok ? "ckan-panel__status--ok" : ""}`}>
          <span aria-hidden="true" />
          {status.message}
        </span>
      </div>
      <div className="ckan-panel__controls">
        <input
          className="ckan-panel__input"
          type="url"
          value={endpoint}
          onChange={(event) => setEndpoint(event.target.value)}
          placeholder="https://opendata.muenchen.de/"
          aria-label="CKAN endpoint URL"
        />
        <button className="ckan-panel__button ckan-panel__button--primary" type="button" onClick={connect} disabled={isConnecting}>
          {isConnecting ? "Connecting" : "Connect"}
        </button>
        <button className="ckan-panel__button" type="button" onClick={reset} disabled={isConnecting}>
          Reset
        </button>
      </div>
    </section>
  );
}

function LlmRolesPanel() {
  const [roles, setRoles] = useState([]);
  const [runtimeStatus, setRuntimeStatus] = useState(null);
  const [isValidating, setIsValidating] = useState(false);
  const [message, setMessage] = useState("Loading model roles...");

  const loadStatus = () => {
    Promise.all([
      fetch("/api/llms/status").then((response) => response.json()),
      fetch("/api/minicpm/status").then((response) => response.json()),
    ])
      .then(([llmData, minicpmData]) => {
        setRoles(llmData.roles || []);
        setRuntimeStatus(minicpmData);
        setMessage(`${minicpmData.backend || "MiniCPM"} role configuration.`);
      })
      .catch(() => {
        setMessage("Could not load model role status.");
      });
  };

  useEffect(() => {
    loadStatus();
  }, []);

  const validate = async () => {
    setIsValidating(true);
    setMessage("Validating OpenAI-compatible providers...");
    try {
      const response = await fetch("/api/llms/validate", { method: "POST" });
      const data = await response.json();
      setRoles(data.roles || []);
      setMessage("Validation complete.");
    } catch {
      setMessage("Could not validate LLM roles.");
    } finally {
      setIsValidating(false);
    }
  };

  return (
    <section className="llm-panel" aria-label="Model role configuration">
      <div className="llm-panel__header">
        <div>
          <span className="llm-panel__label">Model roles</span>
          <span className="llm-panel__message">{message}</span>
        </div>
        <button className="ckan-panel__button ckan-panel__button--primary" type="button" onClick={validate} disabled={isValidating}>
          {isValidating ? "Validating" : "Validate"}
        </button>
      </div>
      <div className="llm-panel__roles">
        {roles.map((role) => (
          <div className="llm-role" key={role.key}>
            <span className={`llm-role__dot llm-role__dot--${role.validation_status || "missing"}`} aria-hidden="true" />
            <div className="llm-role__body">
              <span className="llm-role__name">{role.label}</span>
              <span className="llm-role__links">
                {runtimeForRole(runtimeStatus, role.key)?.model_hub_url ? (
                  <a href={runtimeForRole(runtimeStatus, role.key).model_hub_url} target="_blank" rel="noreferrer">
                    Base: {modelArtifactLabel(
                      runtimeForRole(runtimeStatus, role.key)?.model_repo_id,
                      runtimeForRole(runtimeStatus, role.key)?.model_filename,
                      role.model || "No model",
                    )}
                  </a>
                ) : (
                  <span>
                    Base: {modelArtifactLabel(
                      runtimeForRole(runtimeStatus, role.key)?.model_repo_id,
                      runtimeForRole(runtimeStatus, role.key)?.model_filename,
                      role.model || "No model",
                    )}
                  </span>
                )}
                {runtimeForRole(runtimeStatus, role.key)?.lora_hub_url ? (
                  <a href={runtimeForRole(runtimeStatus, role.key).lora_hub_url} target="_blank" rel="noreferrer">
                    Adapter: {modelArtifactLabel(
                      runtimeForRole(runtimeStatus, role.key)?.lora_repo_id,
                      runtimeForRole(runtimeStatus, role.key)?.lora_filename,
                      "Role adapter",
                    )}
                  </a>
                ) : null}
              </span>
              <span className="llm-role__meta">{role.description}</span>
              <span className="llm-role__status">{role.message}</span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function TracePanel({ traceId }) {
  const [trace, setTrace] = useState(null);
  const [message, setMessage] = useState("No trace yet.");

  const loadTrace = async (id = traceId) => {
    const url = id ? `/api/traces/${encodeURIComponent(id)}` : "/api/traces/latest?limit=1";
    try {
      const response = await fetch(url);
      const data = await response.json();
      const nextTrace = id ? data : data.traces?.[0] || null;
      if (nextTrace?.error) {
        setMessage("Trace not found.");
        return;
      }
      setTrace(nextTrace);
      setMessage(nextTrace ? "Latest model trace." : "No trace yet.");
    } catch {
      setMessage("Could not load trace.");
    }
  };

  useEffect(() => {
    loadTrace(traceId);
  }, [traceId]);

  const events = trace?.events || [];
  const runtime = trace?.runtime || {};
  const cache = trace?.cache || {};
  const modelLabel = runtime.model_filename || runtime.model_path || runtime.model_repo_id || trace?.model_family || "unconfigured";

  return (
    <section className="trace-panel" aria-label="Model trace">
      <div className="trace-panel__header">
        <div>
          <span className="trace-panel__label">Trace</span>
          <span className="trace-panel__message">{message}</span>
        </div>
        <button className="ckan-panel__button" type="button" onClick={() => loadTrace()}>
          Refresh
        </button>
      </div>
      {trace ? (
        <>
          <div className="trace-panel__summary">
            <span>{trace.backend}</span>
            <span>{trace.role}</span>
            <span>{cache.hit === true ? "cache hit" : cache.hit === false ? "cache miss" : "fallback"}</span>
            <span>{trace.duration_ms ?? 0} ms</span>
          </div>
          <div className="trace-panel__meta">
            <span>{modelLabel}</span>
            {runtime.lora_filename || runtime.lora_path ? <span>{runtime.lora_filename || runtime.lora_path}</span> : null}
          </div>
          <ol className="trace-panel__events">
            {events.map((event, index) => (
              <li className="trace-event" key={`${event.name}-${index}`}>
                <span className="trace-event__name">{event.name}</span>
                <span className="trace-event__detail">{event.detail}</span>
              </li>
            ))}
          </ol>
        </>
      ) : null}
    </section>
  );
}

function BackendConfigHeader({ onCkanConnectionChange, traceId }) {
  return (
    <div className="backend-config">
      <CkanEndpointPanel onConnectionChange={onCkanConnectionChange} />
      <LlmRolesPanel />
      <TracePanel traceId={traceId} />
    </div>
  );
}

function App() {
  const [ckanConnection, setCkanConnection] = useState({ connected: false, base_url: "https://opendata.muenchen.de/" });
  const [traceId, setTraceId] = useState(null);

  const processMessage = async ({ threadId, messages, abortController }) => {
    console.info("[smolnalysis] sending chat request", {
      threadId,
      adapter: CHAT_ADAPTER,
      messageCount: messages?.length || 0,
      ckan: ckanConnection,
    });

    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        threadId,
        messages,
        adapter: CHAT_ADAPTER,
        ckan: ckanConnection,
      }),
      signal: abortController.signal,
    });
    const nextTraceId = response.headers.get("x-smolnalysis-trace-id");
    if (nextTraceId) {
      setTraceId(nextTraceId);
    }
    return response;
  };

  return (
    <FullScreen
      processMessage={processMessage}
      streamProtocol={openAIAdapter()}
      componentLibrary={openuiChatLibrary}
      agentName="smolnalysis"
      logoUrl="/static/smolnalysis-mark.svg"
      showAssistantLogo={false}
      threadHeader={<BackendConfigHeader onCkanConnectionChange={setCkanConnection} traceId={traceId} />}
      welcomeMessage={{
        title: "smolnalysis",
        description: "Ask about the demo dataset and receive mocked OpenUI-Lang responses.",
      }}
      conversationStarters={{
        variant: "short",
        options: [
          { displayText: "Summarize", prompt: "Summarize this dataset" },
          { displayText: "Schema", prompt: "List the columns and missing values" },
          { displayText: "Bar chart", prompt: "Show a bar chart of population by city" },
          { displayText: "Histogram", prompt: "Show a histogram of median_age" },
        ],
      }}
    />
  );
}

createRoot(document.getElementById("root")).render(<App />);
