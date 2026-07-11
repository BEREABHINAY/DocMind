import { useState, useEffect, useRef } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function ScoreDial({ score }) {
  // score is a cosine similarity in [0,1]-ish range; clamp for display
  const pct = Math.max(0, Math.min(1, score));
  return (
    <div className="score-dial" title={`relevance ${(pct * 100).toFixed(0)}%`}>
      <svg viewBox="0 0 36 36" className="score-dial-svg">
        <path
          className="score-dial-track"
          d="M18 2 a 16 16 0 0 1 0 32 a 16 16 0 0 1 0 -32"
        />
        <path
          className="score-dial-fill"
          strokeDasharray={`${pct * 100}, 100`}
          d="M18 2 a 16 16 0 0 1 0 32 a 16 16 0 0 1 0 -32"
        />
      </svg>
      <span className="score-dial-label">{(pct * 100).toFixed(0)}</span>
    </div>
  );
}

function SourceCard({ source }) {
  return (
    <div className="source-card">
      <div className="source-card-mark" />
      <div className="source-card-body">
        <div className="source-card-head">
          <span className="source-card-name">{source.source}</span>
          <ScoreDial score={source.score} />
        </div>
        <p className="source-card-text">{source.text}</p>
      </div>
    </div>
  );
}

function Message({ message }) {
  if (message.role === "user") {
    return (
      <div className="msg msg-user">
        <div className="msg-bubble">{message.text}</div>
      </div>
    );
  }
  return (
    <div className="msg msg-assistant">
      <div className="msg-bubble">
        <p className="msg-answer">{message.text}</p>
        {message.sources && message.sources.length > 0 && (
          <details className="sources-toggle" open={message.sources.length <= 2}>
            <summary>
              {message.sources.length} passage{message.sources.length > 1 ? "s" : ""} referenced
            </summary>
            <div className="sources-list">
              {message.sources.map((s, i) => (
                <SourceCard key={i} source={s} />
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [isAsking, setIsAsking] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [status, setStatus] = useState(null);
  const [apiOnline, setApiOnline] = useState(null);
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  const refreshDocuments = async () => {
    try {
      const res = await fetch(`${API_URL}/documents`);
      if (!res.ok) throw new Error();
      setDocuments(await res.json());
      setApiOnline(true);
    } catch {
      setApiOnline(false);
    }
  };

  useEffect(() => {
    refreshDocuments();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleUpload = async (file) => {
    if (!file) return;
    setIsUploading(true);
    setStatus(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${API_URL}/ingest`, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      setStatus({ type: "ok", text: data.message });
      await refreshDocuments();
    } catch (err) {
      setStatus({ type: "error", text: err.message });
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDelete = async (source) => {
    await fetch(`${API_URL}/documents/${encodeURIComponent(source)}`, { method: "DELETE" });
    await refreshDocuments();
  };

  const handleAsk = async (e) => {
    e.preventDefault();
    const q = question.trim();
    if (!q || isAsking) return;

    setMessages((prev) => [...prev, { role: "user", text: q }]);
    setQuestion("");
    setIsAsking(true);

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Query failed");
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: data.answer, sources: data.sources },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `Something went wrong: ${err.message}`, sources: [] },
      ]);
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">§</span>
          <div>
            <h1>DocuMind</h1>
            <p className="brand-tagline">ask your documents directly</p>
          </div>
        </div>

        <div className={`api-status ${apiOnline === false ? "offline" : ""}`}>
          <span className="api-status-dot" />
          {apiOnline === null ? "checking backend…" : apiOnline ? "backend connected" : "backend unreachable"}
        </div>

        <div className="upload-zone">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.md"
            id="file-upload"
            onChange={(e) => handleUpload(e.target.files[0])}
            hidden
          />
          <label htmlFor="file-upload" className={`upload-button ${isUploading ? "busy" : ""}`}>
            {isUploading ? "indexing…" : "+ add a document"}
          </label>
          {status && <p className={`upload-status ${status.type}`}>{status.text}</p>}
        </div>

        <div className="doc-library">
          <h2>library</h2>
          {documents.length === 0 ? (
            <p className="doc-empty">No documents indexed yet. Add a PDF or text file to begin.</p>
          ) : (
            <ul>
              {documents.map((doc) => (
                <li key={doc.source} className="doc-item">
                  <div>
                    <span className="doc-name">{doc.source}</span>
                    <span className="doc-meta">{doc.chunk_count} chunks</span>
                  </div>
                  <button
                    className="doc-remove"
                    onClick={() => handleDelete(doc.source)}
                    aria-label={`remove ${doc.source}`}
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      <main className="chat-pane">
        <div className="chat-scroll">
          {messages.length === 0 ? (
            <div className="empty-state">
              <span className="empty-state-mark">§</span>
              <h2>Ask a question about your documents</h2>
              <p>
                Add a document on the left, then ask anything — the answer is grounded
                in what's actually in your files, with the exact passages shown below it.
              </p>
            </div>
          ) : (
            messages.map((m, i) => <Message key={i} message={m} />)
          )}
          {isAsking && (
            <div className="msg msg-assistant">
              <div className="msg-bubble thinking">
                <span className="dot" />
                <span className="dot" />
                <span className="dot" />
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <form className="ask-bar" onSubmit={handleAsk}>
          <input
            type="text"
            placeholder="What's the minimum aggregate required for placements?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={isAsking}
          />
          <button type="submit" disabled={isAsking || !question.trim()}>
            ask
          </button>
        </form>
      </main>
    </div>
  );
}
