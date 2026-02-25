const state = {
  messages: [],
  streamAbort: null,
  lastRequest: null,
  lastResponse: null,
  lastRetrieval: [],
  serverStatus: null,
  serverAction: null,
  trainStatus: null,
  ragStatus: null,
};

const el = {
  chatFeed: document.getElementById("chatFeed"),
  sendBtn: document.getElementById("sendBtn"),
  stopBtn: document.getElementById("stopBtn"),
  clearChatBtn: document.getElementById("clearChatBtn"),
  saveSessionBtn: document.getElementById("saveSessionBtn"),
  loadSessionInput: document.getElementById("loadSessionInput"),
  userInput: document.getElementById("userInput"),
  systemPromptInput: document.getElementById("systemPromptInput"),
  ragEnabled: document.getElementById("ragEnabled"),
  ragTopKInput: document.getElementById("ragTopKInput"),
  modelPathInput: document.getElementById("modelPathInput"),
  serverHostInput: document.getElementById("serverHostInput"),
  serverPortInput: document.getElementById("serverPortInput"),
  threadsInput: document.getElementById("threadsInput"),
  ctxSizeInput: document.getElementById("ctxSizeInput"),
  temperatureInput: document.getElementById("temperatureInput"),
  topPInput: document.getElementById("topPInput"),
  maxTokensInput: document.getElementById("maxTokensInput"),
  repeatPenaltyInput: document.getElementById("repeatPenaltyInput"),
  seedInput: document.getElementById("seedInput"),
  startServerBtn: document.getElementById("startServerBtn"),
  stopServerBtn: document.getElementById("stopServerBtn"),
  healthBadge: document.getElementById("healthBadge"),
  llamaBadge: document.getElementById("llamaBadge"),
  ragPathsInput: document.getElementById("ragPathsInput"),
  ragUploadInput: document.getElementById("ragUploadInput"),
  uploadRagBtn: document.getElementById("uploadRagBtn"),
  chunkSizeInput: document.getElementById("chunkSizeInput"),
  chunkOverlapInput: document.getElementById("chunkOverlapInput"),
  indexRagBtn: document.getElementById("indexRagBtn"),
  resetRagBtn: document.getElementById("resetRagBtn"),
  ragStatus: document.getElementById("ragStatus"),
  inspector: document.getElementById("inspector"),
  trainBaseModelInput: document.getElementById("trainBaseModelInput"),
  trainUploadInput: document.getElementById("trainUploadInput"),
  uploadTrainBtn: document.getElementById("uploadTrainBtn"),
  trainRawDataInput: document.getElementById("trainRawDataInput"),
  trainRawFormatInput: document.getElementById("trainRawFormatInput"),
  trainRawNameInput: document.getElementById("trainRawNameInput"),
  createRawDatasetBtn: document.getElementById("createRawDatasetBtn"),
  trainDatasetInput: document.getElementById("trainDatasetInput"),
  trainOutputDirInput: document.getElementById("trainOutputDirInput"),
  trainEpochsInput: document.getElementById("trainEpochsInput"),
  trainBatchInput: document.getElementById("trainBatchInput"),
  trainGradAccumInput: document.getElementById("trainGradAccumInput"),
  trainLrInput: document.getElementById("trainLrInput"),
  trainMaxSeqInput: document.getElementById("trainMaxSeqInput"),
  trainLoraRankInput: document.getElementById("trainLoraRankInput"),
  startTrainBtn: document.getElementById("startTrainBtn"),
  stopTrainBtn: document.getElementById("stopTrainBtn"),
  trainStatus: document.getElementById("trainStatus"),
  tabButtons: Array.from(document.querySelectorAll(".tab-btn")),
  tabPanels: Array.from(document.querySelectorAll(".tab-panel")),
};

function pretty(obj) {
  try {
    return JSON.stringify(obj, null, 2);
  } catch (err) {
    return String(obj);
  }
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function updateInspector() {
  el.inspector.textContent = pretty({
    server: state.serverStatus,
    rag: state.ragStatus,
    train: state.trainStatus,
    retrieval: state.lastRetrieval,
    lastRequest: state.lastRequest,
    lastResponse: state.lastResponse,
  });
}

function syncServerButtons(isRunning) {
  if (state.serverAction === "starting") {
    el.startServerBtn.hidden = false;
    el.startServerBtn.disabled = true;
    el.startServerBtn.textContent = "Starting server...";
    el.stopServerBtn.hidden = true;
    return;
  }

  if (state.serverAction === "stopping") {
    el.stopServerBtn.hidden = false;
    el.stopServerBtn.disabled = true;
    el.stopServerBtn.textContent = "Stopping server...";
    el.startServerBtn.hidden = true;
    return;
  }

  el.startServerBtn.textContent = "Start Server";
  el.stopServerBtn.textContent = "Stop Server";
  el.startServerBtn.disabled = false;
  el.stopServerBtn.disabled = false;
  el.startServerBtn.hidden = !!isRunning;
  el.stopServerBtn.hidden = !isRunning;
}

function setActiveTab(tabName) {
  for (const button of el.tabButtons) {
    button.classList.toggle("active", button.dataset.tab === tabName);
  }
  for (const panel of el.tabPanels) {
    panel.classList.toggle("active", panel.dataset.panel === tabName);
  }
}

function renderMessages() {
  el.chatFeed.innerHTML = "";
  for (const msg of state.messages) {
    const node = document.createElement("article");
    node.className = "msg";
    node.dataset.role = msg.role;
    const role = document.createElement("header");
    role.className = "msg-role";
    role.textContent = msg.role;
    const body = document.createElement("div");
    body.className = "msg-body";
    body.innerHTML = escapeHtml(msg.content || "");
    node.appendChild(role);
    node.appendChild(body);
    el.chatFeed.appendChild(node);
  }
  el.chatFeed.scrollTop = el.chatFeed.scrollHeight;
}

function pushMessage(role, content) {
  state.messages.push({ role, content });
  renderMessages();
  return state.messages.length - 1;
}

async function callJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = { raw: await response.text() };
  }
  if (!response.ok) {
    const detail = payload?.detail || payload?.error || JSON.stringify(payload);
    throw new Error(detail);
  }
  return payload;
}

async function refreshHealth() {
  try {
    const health = await callJson("/api/health");
    const serverStatus = await callJson("/api/server/status");
    state.serverStatus = serverStatus;
    if (!state.serverAction) {
      syncServerButtons(!!serverStatus.running);
    }
    el.healthBadge.textContent = "Web UI ready";
    el.healthBadge.className = "badge good";
    if (health.llama_running) {
      el.llamaBadge.textContent = health.llama_healthy ? "Llama healthy" : "Llama starting";
      el.llamaBadge.className = `badge ${health.llama_healthy ? "good" : "muted"}`;
    } else {
      el.llamaBadge.textContent = "Llama stopped";
      el.llamaBadge.className = "badge bad";
    }
  } catch (err) {
    if (!state.serverAction) {
      syncServerButtons(false);
    }
    el.healthBadge.textContent = "Web UI error";
    el.healthBadge.className = "badge bad";
    el.llamaBadge.textContent = String(err.message || err);
    el.llamaBadge.className = "badge bad";
  } finally {
    updateInspector();
  }
}

async function refreshRag() {
  try {
    const rag = await callJson("/api/rag/status");
    state.ragStatus = rag;
    el.ragStatus.textContent = pretty(rag);
  } catch (err) {
    el.ragStatus.textContent = `RAG status error: ${err.message || err}`;
  } finally {
    updateInspector();
  }
}

async function refreshTraining() {
  try {
    const status = await callJson("/api/train/status");
    const logs = await callJson("/api/train/logs?limit=80");
    state.trainStatus = status;
    el.trainStatus.textContent = `${pretty(status)}\n\n--- logs ---\n${(logs.logs || []).join("\n")}`;
  } catch (err) {
    el.trainStatus.textContent = `Training status error: ${err.message || err}`;
  } finally {
    updateInspector();
  }
}

function buildChatPayload() {
  const seedRaw = el.seedInput.value.trim();
  return {
    messages: state.messages.filter((m) => m.role === "user" || m.role === "assistant"),
    system_prompt: el.systemPromptInput.value,
    temperature: Number(el.temperatureInput.value || 0.8),
    top_p: Number(el.topPInput.value || 0.95),
    max_tokens: Number(el.maxTokensInput.value || 512),
    repeat_penalty: Number(el.repeatPenaltyInput.value || 1.05),
    seed: seedRaw ? Number(seedRaw) : null,
    rag_enabled: !!el.ragEnabled.checked,
    rag_top_k: Number(el.ragTopKInput.value || 4),
    stop: [],
  };
}

function parseSseBlock(block, assistantIndex) {
  if (!block.trim()) return;
  const lines = block.split("\n");
  let eventName = "message";
  const dataLines = [];
  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  const dataText = dataLines.join("\n");
  if (!dataText) return;

  if (eventName === "retrieval") {
    try {
      const payload = JSON.parse(dataText);
      state.lastRetrieval = Array.isArray(payload?.retrieval) ? payload.retrieval : [];
      state.lastResponse = { ...(state.lastResponse || {}), retrieval: state.lastRetrieval };
      updateInspector();
    } catch {
      // ignore parsing issue for retrieval side-channel
    }
    return;
  }

  if (eventName === "error") {
    throw new Error(dataText);
  }

  if (dataText === "[DONE]") {
    return;
  }

  let parsed;
  try {
    parsed = JSON.parse(dataText);
  } catch {
    return;
  }

  const choice = parsed?.choices?.[0] || {};
  const delta = choice?.delta?.content ?? "";
  if (delta) {
    state.messages[assistantIndex].content += delta;
    renderMessages();
  }

  if (parsed?.usage) {
    state.lastResponse = { ...(state.lastResponse || {}), usage: parsed.usage };
    updateInspector();
  }
}

function looksLikeRagRefusal(text) {
  const normalized = String(text || "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
  if (!normalized) return false;

  const patterns = [
    "can't access",
    "cannot access",
    "can't directly",
    "cannot directly",
    "unable to directly",
    "can't retrieve",
    "cannot retrieve",
    "can't view",
    "cannot view",
    "can't analyze",
    "cannot analyze",
    "if you provide the text",
  ];
  const mentionsDoc = /file|files|document|documents|attachment|attached|scan|image|paper/.test(normalized);
  return mentionsDoc && patterns.some((pattern) => normalized.includes(pattern));
}

function tokenizeForFallback(text) {
  return String(text || "")
    .toLowerCase()
    .match(/[a-z0-9_]+/g) || [];
}

function buildExtractiveFallback(query, chunks) {
  const stop = new Set([
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "if", "in", "is", "it", "its",
    "me", "my", "of", "on", "or", "our", "please", "said", "she", "so", "that", "the", "their",
    "them", "there", "they", "this", "to", "us", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "with", "would", "you", "your",
  ]);
  const queryTokens = tokenizeForFallback(query).filter((token) => !stop.has(token));
  const scored = [];

  for (const chunk of chunks || []) {
    const source = String(chunk?.source || "unknown-source");
    const sentences = String(chunk?.text || "")
      .split(/(?<=[.!?])\s+|\n+/)
      .map((s) => s.trim())
      .filter(Boolean);
    for (const sentence of sentences) {
      const words = new Set(tokenizeForFallback(sentence));
      if (words.size === 0) continue;
      let score = 0;
      for (const token of queryTokens) {
        if (words.has(token)) score += 1;
      }
      scored.push({ score, sentence, source });
    }
  }

  let selected = [];
  if (scored.length > 0) {
    scored.sort((a, b) => b.score - a.score);
    selected = scored.slice(0, 3);
  }

  if (selected.length === 0) {
    const firstChunk = chunks?.[0];
    if (!firstChunk) return "";
    const src = String(firstChunk.source || "unknown-source");
    const snippet = String(firstChunk.text || "").replace(/\s+/g, " ").slice(0, 600);
    return `From the indexed document(s):\n- ${snippet} [${src}]`;
  }

  const lines = ["From the indexed document(s):"];
  for (const item of selected) {
    lines.push(`- ${item.sentence} [${item.source}]`);
  }
  return lines.join("\n");
}

async function applyFrontendRagFallback(assistantIndex, userQuery) {
  if (!el.ragEnabled.checked) return;
  const content = String(state.messages?.[assistantIndex]?.content || "");
  if (!looksLikeRagRefusal(content)) return;

  let chunks = Array.isArray(state.lastRetrieval) ? state.lastRetrieval : [];
  if (chunks.length === 0) {
    try {
      const searchResult = await callJson("/api/rag/search", {
        method: "POST",
        body: JSON.stringify({
          query: userQuery,
          top_k: Number(el.ragTopKInput.value || 4),
        }),
      });
      chunks = Array.isArray(searchResult?.results) ? searchResult.results : [];
    } catch {
      chunks = [];
    }
  }
  if (chunks.length === 0) return;

  const fallback = buildExtractiveFallback(userQuery, chunks);
  if (!fallback) return;

  state.lastRetrieval = chunks;
  state.lastResponse = { ...(state.lastResponse || {}), frontend_rag_fallback: true, retrieval: chunks };
  state.messages[assistantIndex].content = fallback;
  renderMessages();
  updateInspector();
}

async function sendMessage() {
  const text = el.userInput.value.trim();
  if (!text) return;
  if (state.streamAbort) return;

  pushMessage("user", text);
  const assistantIndex = pushMessage("assistant", "");
  el.userInput.value = "";

  const payload = buildChatPayload();
  state.lastRequest = payload;
  state.lastResponse = null;
  updateInspector();

  const controller = new AbortController();
  state.streamAbort = controller;
  el.sendBtn.disabled = true;
  el.stopBtn.disabled = false;

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok || !response.body) {
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let splitIndex = buffer.indexOf("\n\n");
      while (splitIndex >= 0) {
        const block = buffer.slice(0, splitIndex);
        buffer = buffer.slice(splitIndex + 2);
        parseSseBlock(block, assistantIndex);
        splitIndex = buffer.indexOf("\n\n");
      }
    }
  } catch (err) {
    state.messages[assistantIndex].content += `\n\n[stream error] ${err.message || err}`;
    renderMessages();
  } finally {
    await applyFrontendRagFallback(assistantIndex, text);
    state.streamAbort = null;
    el.sendBtn.disabled = false;
    el.stopBtn.disabled = true;
    updateInspector();
  }
}

async function startServer() {
  if (state.serverAction) return;
  state.serverAction = "starting";
  syncServerButtons(false);
  try {
    const payload = {
      model_path: el.modelPathInput.value.trim(),
      host: el.serverHostInput.value.trim() || "127.0.0.1",
      port: Number(el.serverPortInput.value || 8080),
      threads: Number(el.threadsInput.value || 8),
      ctx_size: Number(el.ctxSizeInput.value || 2048),
      n_predict: 4096,
      lora_paths: [],
      extra_args: [],
    };
    const result = await callJson("/api/server/start", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.serverStatus = result;
  } catch (err) {
    alert(`Server start failed: ${err.message || err}`);
  } finally {
    state.serverAction = null;
    await refreshHealth();
  }
}

async function stopServer() {
  if (state.serverAction) return;
  state.serverAction = "stopping";
  syncServerButtons(true);
  try {
    const result = await callJson("/api/server/stop", { method: "POST" });
    state.serverStatus = result;
  } catch (err) {
    alert(`Server stop failed: ${err.message || err}`);
  } finally {
    state.serverAction = null;
    await refreshHealth();
  }
}

async function indexRag() {
  const paths = el.ragPathsInput.value
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
  if (paths.length === 0) {
    alert("Enter at least one file or directory path.");
    return;
  }
  try {
    const payload = {
      paths,
      chunk_size: Number(el.chunkSizeInput.value || 220),
      chunk_overlap: Number(el.chunkOverlapInput.value || 40),
      reset: false,
    };
    const result = await callJson("/api/rag/index", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.lastResponse = result;
    await refreshRag();
  } catch (err) {
    alert(`RAG indexing failed: ${err.message || err}`);
  }
}

function mergeUniquePaths(existingText, newPaths) {
  const merged = new Set(
    existingText
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
  );
  for (const path of newPaths) {
    if (path && String(path).trim()) {
      merged.add(String(path).trim());
    }
  }
  return Array.from(merged).join("\n");
}

async function uploadRagFiles() {
  const files = Array.from(el.ragUploadInput.files || []);
  if (files.length === 0) {
    alert("Select at least one file to upload.");
    return;
  }

  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  formData.append("chunk_size", String(Number(el.chunkSizeInput.value || 220)));
  formData.append("chunk_overlap", String(Number(el.chunkOverlapInput.value || 40)));
  formData.append("reset", "false");

  el.uploadRagBtn.disabled = true;
  try {
    const response = await fetch("/api/rag/upload", {
      method: "POST",
      body: formData,
    });
    const raw = await response.text();
    let payload = {};
    try {
      payload = raw ? JSON.parse(raw) : {};
    } catch {
      payload = { raw };
    }

    if (!response.ok) {
      const detail = payload?.detail || payload?.error || raw || `HTTP ${response.status}`;
      throw new Error(typeof detail === "string" ? detail : pretty(detail));
    }

    const savedFiles = Array.isArray(payload.saved_files) ? payload.saved_files : [];
    const skipped = Array.isArray(payload.skipped_files) ? payload.skipped_files : [];
    el.ragPathsInput.value = mergeUniquePaths(el.ragPathsInput.value, savedFiles);
    el.ragUploadInput.value = "";

    state.lastResponse = payload;
    await refreshRag();
    const addedChunks = Number(payload.indexed?.added_chunks || 0);
    const skippedSummary =
      skipped.length > 0 ? ` Skipped: ${skipped.map((item) => `${item.file} (${item.reason})`).join(", ")}.` : "";
    if (addedChunks <= 0) {
      alert(
        `Upload completed but no RAG chunks were indexed. Uploaded ${payload.saved_count || 0} file(s).${skippedSummary}`
      );
    } else {
      alert(`Uploaded ${payload.saved_count || 0} file(s). Added ${addedChunks} chunk(s).${skippedSummary}`);
    }
  } catch (err) {
    alert(`RAG upload failed: ${err.message || err}`);
  } finally {
    el.uploadRagBtn.disabled = false;
    updateInspector();
  }
}

async function resetRag() {
  if (!confirm("Reset the RAG index?")) return;
  try {
    const result = await callJson("/api/rag/reset", { method: "POST" });
    state.lastResponse = result;
    await refreshRag();
  } catch (err) {
    alert(`RAG reset failed: ${err.message || err}`);
  }
}

async function startTraining() {
  try {
    const payload = {
      base_model: el.trainBaseModelInput.value.trim(),
      dataset_path: el.trainDatasetInput.value.trim(),
      output_dir: el.trainOutputDirInput.value.trim(),
      epochs: Number(el.trainEpochsInput.value || 1),
      batch_size: Number(el.trainBatchInput.value || 1),
      grad_accum_steps: Number(el.trainGradAccumInput.value || 8),
      learning_rate: Number(el.trainLrInput.value || 0.0002),
      max_seq_len: Number(el.trainMaxSeqInput.value || 1024),
      lora_rank: Number(el.trainLoraRankInput.value || 16),
      lora_alpha: 32,
    };
    const result = await callJson("/api/train/start", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.trainStatus = result;
    updateInspector();
    await refreshTraining();
  } catch (err) {
    alert(`Training start failed: ${err.message || err}`);
  }
}

async function uploadTrainingDataset() {
  const file = el.trainUploadInput.files?.[0];
  if (!file) {
    alert("Select a dataset file first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  el.uploadTrainBtn.disabled = true;
  try {
    const response = await fetch("/api/train/upload-dataset", {
      method: "POST",
      body: formData,
    });
    const raw = await response.text();
    let payload = {};
    try {
      payload = raw ? JSON.parse(raw) : {};
    } catch {
      payload = { raw };
    }
    if (!response.ok) {
      const detail = payload?.detail || payload?.error || raw || `HTTP ${response.status}`;
      throw new Error(typeof detail === "string" ? detail : pretty(detail));
    }

    el.trainDatasetInput.value = payload.dataset_path || "";
    state.lastResponse = payload;
    updateInspector();
    alert(`Uploaded dataset: ${payload.filename}`);
  } catch (err) {
    alert(`Dataset upload failed: ${err.message || err}`);
  } finally {
    el.uploadTrainBtn.disabled = false;
  }
}

async function createTrainingDatasetFromRaw() {
  const rawText = el.trainRawDataInput.value || "";
  if (!rawText.trim()) {
    alert("Enter raw training data first.");
    return;
  }

  const formData = new FormData();
  formData.append("raw_text", rawText);
  formData.append("format", el.trainRawFormatInput.value || "lines");
  formData.append("filename", (el.trainRawNameInput.value || "raw_dataset").trim());

  el.createRawDatasetBtn.disabled = true;
  try {
    const response = await fetch("/api/train/raw-dataset", {
      method: "POST",
      body: formData,
    });
    const raw = await response.text();
    let payload = {};
    try {
      payload = raw ? JSON.parse(raw) : {};
    } catch {
      payload = { raw };
    }
    if (!response.ok) {
      const detail = payload?.detail || payload?.error || raw || `HTTP ${response.status}`;
      throw new Error(typeof detail === "string" ? detail : pretty(detail));
    }

    el.trainDatasetInput.value = payload.dataset_path || "";
    state.lastResponse = payload;
    updateInspector();
    alert(`Prepared dataset: ${payload.filename} (${payload.rows || 0} rows)`);
  } catch (err) {
    alert(`Raw dataset prep failed: ${err.message || err}`);
  } finally {
    el.createRawDatasetBtn.disabled = false;
  }
}

async function stopTraining() {
  try {
    const result = await callJson("/api/train/stop", { method: "POST" });
    state.trainStatus = result;
    updateInspector();
    await refreshTraining();
  } catch (err) {
    alert(`Training stop failed: ${err.message || err}`);
  }
}

function saveSession() {
  const payload = {
    systemPrompt: el.systemPromptInput.value,
    messages: state.messages,
    settings: {
      temperature: el.temperatureInput.value,
      top_p: el.topPInput.value,
      max_tokens: el.maxTokensInput.value,
      repeat_penalty: el.repeatPenaltyInput.value,
      rag_enabled: el.ragEnabled.checked,
      rag_top_k: el.ragTopKInput.value,
    },
    createdAt: new Date().toISOString(),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "bitnet-session.json";
  a.click();
  URL.revokeObjectURL(url);
}

function loadSession(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const payload = JSON.parse(String(reader.result || "{}"));
      state.messages = Array.isArray(payload.messages) ? payload.messages : [];
      el.systemPromptInput.value = payload.systemPrompt || "";
      const settings = payload.settings || {};
      el.temperatureInput.value = settings.temperature ?? el.temperatureInput.value;
      el.topPInput.value = settings.top_p ?? el.topPInput.value;
      el.maxTokensInput.value = settings.max_tokens ?? el.maxTokensInput.value;
      el.repeatPenaltyInput.value = settings.repeat_penalty ?? el.repeatPenaltyInput.value;
      el.ragEnabled.checked = settings.rag_enabled ?? el.ragEnabled.checked;
      el.ragTopKInput.value = settings.rag_top_k ?? el.ragTopKInput.value;
      renderMessages();
    } catch (err) {
      alert(`Invalid session file: ${err.message || err}`);
    }
  };
  reader.readAsText(file);
}

function setupEvents() {
  el.sendBtn.addEventListener("click", sendMessage);
  el.stopBtn.addEventListener("click", () => state.streamAbort && state.streamAbort.abort());
  el.userInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });
  el.startServerBtn.addEventListener("click", startServer);
  el.stopServerBtn.addEventListener("click", stopServer);
  el.indexRagBtn.addEventListener("click", indexRag);
  el.uploadRagBtn.addEventListener("click", uploadRagFiles);
  el.resetRagBtn.addEventListener("click", resetRag);
  el.uploadTrainBtn.addEventListener("click", uploadTrainingDataset);
  el.createRawDatasetBtn.addEventListener("click", createTrainingDatasetFromRaw);
  el.startTrainBtn.addEventListener("click", startTraining);
  el.stopTrainBtn.addEventListener("click", stopTraining);
  el.clearChatBtn.addEventListener("click", () => {
    state.messages = [];
    renderMessages();
  });
  el.saveSessionBtn.addEventListener("click", saveSession);
  el.loadSessionInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) loadSession(file);
  });
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      el.userInput.value = chip.dataset.prompt || "";
      el.userInput.focus();
    });
  });
  for (const button of el.tabButtons) {
    button.addEventListener("click", () => setActiveTab(button.dataset.tab));
  }
}

async function boot() {
  setupEvents();
  syncServerButtons(false);
  await refreshHealth();
  await refreshRag();
  await refreshTraining();
  pushMessage(
    "system",
    "Command Deck ready. Start llama-server, optionally index docs for RAG, then send prompts."
  );
  setInterval(refreshHealth, 5000);
  setInterval(refreshRag, 10000);
  setInterval(refreshTraining, 8000);
}

boot();
