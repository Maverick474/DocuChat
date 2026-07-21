import { useEffect, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ALLOWED_EXTENSIONS = ["pdf", "docx"];

const initialMessages = [
  {
    id: 1,
    role: "assistant",
    text: "Welcome to DocuChat. Upload a PDF or DOCX, then ask a question and I’ll answer only from that document with source citations.",
    citations: [],
  },
];

function getTime() {
  return new Intl.DateTimeFormat([], { hour: "2-digit", minute: "2-digit" }).format(new Date());
}

async function getErrorMessage(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) return data.detail.map((item) => item.msg).join(", ");
  } catch {
    // Use the fallback below when the server does not return JSON.
  }
  return `Request failed with status ${response.status}.`;
}

function App() {
  const [messages, setMessages] = useState(initialMessages);
  const [draft, setDraft] = useState("");
  const [documentInfo, setDocumentInfo] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [serverStatus, setServerStatus] = useState("checking");
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    async function checkServer() {
      try {
        const response = await fetch(`${API_BASE_URL}/health`);
        setServerStatus(response.ok ? "online" : "offline");
      } catch {
        setServerStatus("offline");
      }
    }
    checkServer();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  function validateFile(file) {
    if (!file) return "Please select a file.";
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(extension)) return "Only PDF and DOCX files are supported.";
    if (file.size > MAX_FILE_SIZE) return "The file must be 10 MB or smaller.";
    return "";
  }

  async function uploadFile(file) {
    const validationError = validateFile(file);
    if (validationError) {
      setUploadError(validationError);
      return;
    }

    setUploadError("");
    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, { method: "POST", body: formData });
      if (!response.ok) throw new Error(await getErrorMessage(response));

      const data = await response.json();
      setDocumentInfo({ ...data, extension: file.name.split(".").pop()?.toUpperCase() });
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: `“${data.filename}” is ready. I indexed ${data.chunks_stored} searchable chunk${data.chunks_stored === 1 ? "" : "s"}. What would you like to know?`,
          citations: [],
          time: getTime(),
        },
      ]);
    } catch (error) {
      setUploadError(error.message || "Unable to upload the document.");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    uploadFile(event.dataTransfer.files?.[0]);
  }

  async function sendMessage(event) {
    event?.preventDefault();
    const question = draft.trim();
    if (!question || isSending) return;

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text: question,
      citations: [],
      time: getTime(),
    };
    setMessages((current) => [...current, userMessage]);
    setDraft("");
    setIsSending(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          document_id: documentInfo?.document_id || null,
          top_k: 5,
        }),
      });
      if (!response.ok) throw new Error(await getErrorMessage(response));

      const data = await response.json();
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: data.answer,
          citations: data.citations || [],
          time: getTime(),
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: error.message || "I could not connect to the DocuChat server.",
          citations: [],
          kind: "error",
          time: getTime(),
        },
      ]);
    } finally {
      setIsSending(false);
    }
  }

  function handleComposerKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  function clearConversation() {
    setMessages(initialMessages);
    setDraft("");
  }

  return (
    <>
      <main className="app-shell">
        <aside className="sidebar">
          <div className="brand">
            <div className="brand-mark">DC</div>
            <div>
              <h1>DocuChat</h1>
              <p>Grounded answers from your files</p>
            </div>
          </div>

          <section>
            <p className="side-label">Add a document</p>
            <div
              className={`upload-zone ${isDragging ? "dragging" : ""} ${isUploading ? "busy" : ""}`}
              onDragEnter={(event) => { event.preventDefault(); setIsDragging(true); }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
            >
              <div className="upload-icon">↑</div>
              <strong>{isUploading ? "Processing document…" : "Drop your document here"}</strong>
              <span>PDF or DOCX · maximum 10 MB</span>
              <button className="choose-file" type="button" onClick={() => fileInputRef.current?.click()}>
                {isUploading ? "Please wait" : "Choose file"}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                hidden
                onChange={(event) => uploadFile(event.target.files?.[0])}
              />
            </div>
            {uploadError && <div className="side-error" role="alert">{uploadError}</div>}
          </section>

          {documentInfo && (
            <section>
              <p className="side-label">Active document</p>
              <div className="document-card">
                <div className="document-row">
                  <div className="file-badge">{documentInfo.extension || "FILE"}</div>
                  <div className="document-meta">
                    <div className="document-name" title={documentInfo.filename}>{documentInfo.filename}</div>
                    <div className="document-detail">{documentInfo.chunks_stored} searchable chunks</div>
                  </div>
                  <button className="remove-document" type="button" aria-label="Remove active document" onClick={() => setDocumentInfo(null)}>×</button>
                </div>
              </div>
            </section>
          )}

          <div className="tips">
            <p><strong>Tip:</strong> Ask specific questions about facts, policies, definitions, or summaries contained in your active document.</p>
          </div>

          <div className="server-state">
            <span className={`status-dot ${serverStatus}`}></span>
            {serverStatus === "online" ? "FastAPI server connected" : serverStatus === "offline" ? "FastAPI server unavailable" : "Checking server connection…"}
          </div>
        </aside>

        <section className="chat-panel">
          <header className="topbar">
            <div>
              <h2>{documentInfo ? `Chatting with ${documentInfo.filename}` : "Document assistant"}</h2>
              <p>{documentInfo ? "Answers are limited to the active document" : "Upload a document or query previously indexed files"}</p>
            </div>
            <button className="clear-button" type="button" onClick={clearConversation}>Clear chat</button>
          </header>

          <div className="messages" aria-live="polite">
            <div className="message-list">
              {messages.map((message) => (
                <article key={message.id} className={`message ${message.role} ${message.kind || ""}`}>
                  <div className="avatar">{message.role === "user" ? "YOU" : "AI"}</div>
                  <div className="message-body">
                    <div className="bubble">{message.text}</div>
                    {message.citations?.length > 0 && (
                      <div className="citations" aria-label="Sources">
                        {message.citations.map((citation, index) => (
                          <span className="citation" key={`${citation.source}-${citation.page}-${citation.chunk}-${index}`} title={citation.source}>
                            {citation.source} · page {citation.page} · chunk {citation.chunk}
                          </span>
                        ))}
                      </div>
                    )}
                    {message.time && <div className="message-time">{message.time}</div>}
                  </div>
                </article>
              ))}

              {isSending && (
                <article className="message assistant">
                  <div className="avatar">AI</div>
                  <div className="message-body">
                    <div className="bubble">
                      <span className="typing" aria-label="DocuChat is thinking"><i></i><i></i><i></i></span>
                    </div>
                  </div>
                </article>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          <div className="composer-wrap">
            <form className="composer" onSubmit={sendMessage}>
              <textarea
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                placeholder={documentInfo ? "Ask a question about your document…" : "Ask a question…"}
                rows="1"
                maxLength="2000"
                aria-label="Message"
              />
              <button className="send-button" type="submit" disabled={!draft.trim() || isSending} aria-label="Send message">↑</button>
            </form>
            <p className="composer-note">Enter to send · Shift + Enter for a new line · Responses should be verified against cited sources</p>
          </div>
        </section>
      </main>
    </>
  );
}

export default App;
