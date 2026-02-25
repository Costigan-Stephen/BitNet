const state = {
  messages: [],
  streamAbort: null,
  lastRequest: null,
  lastResponse: null,
  serverStatus: null,
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
  chunkSizeInput: document.getElementById("chunkSizeInput"),
  chunkOverlapInput: document.getElementById("chunkOverlapInput"),
  indexRagBtn: document.getElementById("indexRagBtn"),
  resetRagBtn: document.getElementById("resetRagBtn"),
  ragStatus: document.getElementById("ragStatus"),
  inspector: document.getElementById("inspector"),
  trainBaseModelInput: document.getElementById("trainBaseModelInput"),
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
    lastRequest: state.lastRequest,
    lastResponse: state.lastResponse,
  });
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
      state.lastResponse = payload;
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
    state.streamAbort = null;
    el.sendBtn.disabled = false;
    el.stopBtn.disabled = true;
    updateInspector();
  }
}

async function startServer() {
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
    refreshHealth();
  }
}

async function stopServer() {
  try {
    const result = await callJson("/api/server/stop", { method: "POST" });
    state.serverStatus = result;
  } catch (err) {
    alert(`Server stop failed: ${err.message || err}`);
  } finally {
    refreshHealth();
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
  el.resetRagBtn.addEventListener("click", resetRag);
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
}

async function boot() {
  setupEvents();
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

