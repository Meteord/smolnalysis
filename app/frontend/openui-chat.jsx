import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { openAIAdapter } from "@openuidev/react-headless";
import { FullScreen, openuiChatLibrary } from "@openuidev/react-ui";
import "../../node_modules/@openuidev/react-ui/dist/styles/index.css";

import "./openui-chat.css";

const CKAN_STORAGE_KEY = "smolnalysis.ckanEndpoint";

function CkanEndpointPanel() {
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

function App() {
  return (
    <FullScreen
      apiUrl="/api/chat"
      streamProtocol={openAIAdapter()}
      componentLibrary={openuiChatLibrary}
      agentName="smolnalysis"
      logoUrl="/static/smolnalysis-mark.svg"
      showAssistantLogo={false}
      threadHeader={<CkanEndpointPanel />}
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
