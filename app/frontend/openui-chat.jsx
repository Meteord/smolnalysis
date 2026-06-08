import React from "react";
import { createRoot } from "react-dom/client";
import { openAIAdapter } from "@openuidev/react-headless";
import { FullScreen, openuiChatLibrary } from "@openuidev/react-ui";
import "../../node_modules/@openuidev/react-ui/dist/styles/index.css";

import "./openui-chat.css";

function App() {
  return (
    <FullScreen
      apiUrl="/api/chat"
      streamProtocol={openAIAdapter()}
      componentLibrary={openuiChatLibrary}
      agentName="smolnalysis"
      logoUrl="/static/smolnalysis-mark.svg"
      showAssistantLogo={false}
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
