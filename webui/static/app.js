const state = {
  messages: [],
  conversations: [],
  activeConversationId: "",
  streamAbort: null,
  lastRequest: null,
  lastResponse: null,
  lastRetrieval: [],
  models: null,
  ragFiles: [],
  selectedRagPath: null,
  serverStatus: null,
  serverAction: null,
  trainStatus: null,
  ragStatus: null,
  ragProfiles: null,
  ragProfileDraftMode: true,
  ragSubtab: "files",
  lastTokenUsage: null,
  modelRagDraftLinks: {},
  runtimeModelPath: "",
  runtimeModelSwitching: false,
  workspaceProfiles: [],
  activeWorkspaceProfileId: "",
  settingsSubtab: "runtime",
};

const THEME_STORAGE_KEY = "bitnet-webui-theme";
const SYSTEM_PROMPT_STORAGE_KEY = "bitnet-webui-system-prompt";
const WORKSPACE_PROFILE_STORAGE_KEY = "bitnet-webui-workspace-profiles-v1";
const CONVERSATION_STORAGE_KEY = "bitnet-webui-conversations-v1";
const PROFILE_DEFAULTS = {
  system_prompt: "",
  temperature: 0.8,
  top_p: 0.95,
  max_tokens: 512,
  repeat_penalty: 1.05,
  rag_top_k: 4,
  rag_enabled: true,
  seed: "",
};

const el = {
  themeToggleInput: document.getElementById("themeToggleInput"),
  chatFeed: document.getElementById("chatFeed"),
  sendBtn: document.getElementById("sendBtn"),
  stopBtn: document.getElementById("stopBtn"),
  clearChatBtn: document.getElementById("clearChatBtn"),
  newConversationBtn: document.getElementById("newConversationBtn"),
  conversationList: document.getElementById("conversationList"),
  workspaceProfileSelect: document.getElementById("workspaceProfileSelect"),
  createWorkspaceProfileBtn: document.getElementById("createWorkspaceProfileBtn"),
  saveWorkspaceProfileBtn: document.getElementById("saveWorkspaceProfileBtn"),
  deleteWorkspaceProfileBtn: document.getElementById("deleteWorkspaceProfileBtn"),
  userInput: document.getElementById("userInput"),
  chatRuntimeLabel: document.getElementById("chatRuntimeLabel"),
  modelContextBar: document.getElementById("modelContextBar"),
  systemPromptInput: document.getElementById("systemPromptInput"),
  ragEnabled: document.getElementById("ragEnabled"),
  ragContextToggleWrap: document.getElementById("ragContextToggleWrap"),
  ragContextLabel: document.getElementById("ragContextLabel"),
  tokenUsageBadge: document.getElementById("tokenUsageBadge"),
  ragTopKInput: document.getElementById("ragTopKInput"),
  modelPathInput: document.getElementById("modelPathInput"),
  runtimeModelPathNote: document.getElementById("runtimeModelPathNote"),
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
  ragProfileSelect: document.getElementById("ragProfileSelect"),
  activateRagProfileBtn: document.getElementById("activateRagProfileBtn"),
  newRagProfileBtn: document.getElementById("newRagProfileBtn"),
  saveRagProfileBtn: document.getElementById("saveRagProfileBtn"),
  removeRagProfileBtn: document.getElementById("removeRagProfileBtn"),
  ragProfileNameInput: document.getElementById("ragProfileNameInput"),
  ragProfileDescriptionInput: document.getElementById("ragProfileDescriptionInput"),
  ragProfileSourcesTableBody: document.getElementById("ragProfileSourcesTableBody"),
  addSelectedToProfileBtn: document.getElementById("addSelectedToProfileBtn"),
  removeSelectedFromProfileBtn: document.getElementById("removeSelectedFromProfileBtn"),
  ragProfilesStatus: document.getElementById("ragProfilesStatus"),
  ragFilesTableBody: document.getElementById("ragFilesTableBody"),
  ragFileDetails: document.getElementById("ragFileDetails"),
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
  trainModelRevisionInput: document.getElementById("trainModelRevisionInput"),
  trainTokenizerModelInput: document.getElementById("trainTokenizerModelInput"),
  trainQuantizeOutputInput: document.getElementById("trainQuantizeOutputInput"),
  trainEpochsInput: document.getElementById("trainEpochsInput"),
  trainBatchInput: document.getElementById("trainBatchInput"),
  trainGradAccumInput: document.getElementById("trainGradAccumInput"),
  trainLrInput: document.getElementById("trainLrInput"),
  trainMaxSeqInput: document.getElementById("trainMaxSeqInput"),
  startTrainBtn: document.getElementById("startTrainBtn"),
  stopTrainBtn: document.getElementById("stopTrainBtn"),
  trainGuardNote: document.getElementById("trainGuardNote"),
  trainStatus: document.getElementById("trainStatus"),
  refreshModelsBtn: document.getElementById("refreshModelsBtn"),
  useBaseModelBtn: document.getElementById("useBaseModelBtn"),
  modelProfileNameInput: document.getElementById("modelProfileNameInput"),
  modelProfileDirInput: document.getElementById("modelProfileDirInput"),
  modelProfileComputedPath: document.getElementById("modelProfileComputedPath"),
  browseModelDirBtn: document.getElementById("browseModelDirBtn"),
  modelProfileSetActiveInput: document.getElementById("modelProfileSetActiveInput"),
  modelProfileInitFromBaseInput: document.getElementById("modelProfileInitFromBaseInput"),
  addModelProfileBtn: document.getElementById("addModelProfileBtn"),
  modelsTableBody: document.getElementById("modelsTableBody"),
  modelsStatus: document.getElementById("modelsStatus"),
  tabButtons: Array.from(document.querySelectorAll(".tab-btn")),
  tabPanels: Array.from(document.querySelectorAll(".tab-panel")),
  settingsSubtabButtons: Array.from(document.querySelectorAll(".settings-subtab-btn")),
  settingsSubtabPanels: Array.from(document.querySelectorAll(".settings-subtab-panel")),
  ragSubtabButtons: Array.from(document.querySelectorAll(".rag-subtab-btn")),
  ragSubtabPanels: Array.from(document.querySelectorAll(".rag-subtab-panel")),
};

function formatBytes(value) {
  const size = Number(value || 0);
  if (!Number.isFinite(size) || size <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let current = size;
  let index = 0;
  while (current >= 1024 && index < units.length - 1) {
    current /= 1024;
    index += 1;
  }
  return `${current.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatTime(ts) {
  const num = Number(ts || 0);
  if (!Number.isFinite(num) || num <= 0) return "n/a";
  const d = new Date(num * 1000);
  return d.toLocaleString();
}

function escapeAttr(text) {
  return String(text || "").replaceAll("&", "&amp;").replaceAll("\"", "&quot;").replaceAll("<", "&lt;");
}

function normalizePath(path) {
  return String(path || "").trim().replaceAll("\\", "/").toLowerCase();
}

function datasetStemFromPath(path) {
  const raw = String(path || "");
  const name = raw.split(/[/\\]/).pop() || "rag_dataset";
  const stem = name.replace(/\.[^.]+$/, "");
  const safe = stem.replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^[._-]+|[._-]+$/g, "");
  return safe || "rag_dataset";
}

function profileSlug(name) {
  const safe = String(name || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return safe || "model";
}

function profileFilename(name) {
  return `${profileSlug(name)}.gguf`;
}

function joinPath(dirPath, filename) {
  const base = String(dirPath || "").trim();
  const file = String(filename || "").trim();
  if (!base) return file;
  if (!file) return base;
  if (/^[A-Za-z]:$/.test(base)) {
    return `${base}\\${file}`;
  }
  const trimmed = base.replace(/[\\/]+$/, "");
  const sep = trimmed.includes("\\") ? "\\" : "/";
  return `${trimmed}${sep}${file}`;
}

function dirnamePath(pathValue) {
  const value = String(pathValue || "").trim();
  if (!value) return "";
  const normalized = value.replaceAll("/", "\\");
  const idx = normalized.lastIndexOf("\\");
  if (idx <= 1 && /^[A-Za-z]:\\/.test(normalized)) {
    return normalized.slice(0, 3);
  }
  if (idx <= 0) return "";
  return normalized.slice(0, idx);
}

function statusTone(status) {
  const value = String(status || "").toLowerCase();
  if (value === "indexed" || value === "active" || value === "ready") return "good";
  if (value === "missing" || value === "error") return "bad";
  return "warn";
}

function statusPill(label, tone) {
  return `<span class="status-pill ${escapeAttr(tone || "warn")}">${escapeHtml(String(label || ""))}</span>`;
}

function getModelsPayload(payload) {
  if (payload && typeof payload === "object" && payload.models && typeof payload.models === "object") {
    return payload.models;
  }
  return payload;
}

function getRagProfilesPayload(payload) {
  if (payload && typeof payload === "object" && payload.rag_profiles && typeof payload.rag_profiles === "object") {
    return payload.rag_profiles;
  }
  return null;
}

function getRagProfilesArray() {
  return Array.isArray(state.ragProfiles?.profiles) ? state.ragProfiles.profiles : [];
}

function activeRagProfileId() {
  return String(state.ragProfiles?.active_profile_id || "").trim();
}

function effectiveRagProfileIdForChat() {
  const selectedTopProfileId = String(state.activeWorkspaceProfileId || "").trim();
  if (selectedTopProfileId && findRagProfile(selectedTopProfileId)) {
    return selectedTopProfileId;
  }
  const selectedId = selectedRagProfileId();
  if (findRagProfile(selectedId)) {
    return selectedId;
  }
  const activeWorkspace = findWorkspaceProfile(state.activeWorkspaceProfileId);
  const workspaceLinkedRagId = String(activeWorkspace?.rag_profile_id || "").trim();
  if (workspaceLinkedRagId && findRagProfile(workspaceLinkedRagId)) {
    return workspaceLinkedRagId;
  }
  const activeBackendRagId = activeRagProfileId();
  if (activeBackendRagId && findRagProfile(activeBackendRagId)) {
    return activeBackendRagId;
  }
  return "";
}

function findRagProfile(profileId) {
  const id = String(profileId || "").trim();
  if (!id) return null;
  return getRagProfilesArray().find((item) => String(item.id || "") === id) || null;
}

function chunkCountForSource(pathValue) {
  const key = normalizePath(pathValue);
  if (!key) return 0;
  const match = (state.ragFiles || []).find((item) => normalizePath(item.path) === key);
  return Number(match?.chunk_count || 0);
}

function hasOwn(obj, key) {
  return Object.prototype.hasOwnProperty.call(obj, key);
}

function reconcileModelRagDraftLinks(modelStatus) {
  const profiles = Array.isArray(modelStatus?.profiles) ? modelStatus.profiles : [];
  const backendByProfileId = {};
  for (const profile of profiles) {
    const profileId = String(profile?.id || "");
    if (!profileId) continue;
    backendByProfileId[profileId] = String(profile?.rag_profile_id || "");
  }

  const nextDrafts = {};
  const currentDrafts = state.modelRagDraftLinks || {};
  for (const [profileId, draftValue] of Object.entries(currentDrafts)) {
    if (!hasOwn(backendByProfileId, profileId)) continue;
    const normalizedDraft = String(draftValue || "");
    const normalizedBackend = String(backendByProfileId[profileId] || "");
    if (normalizedDraft !== normalizedBackend) {
      nextDrafts[profileId] = normalizedDraft;
    }
  }
  state.modelRagDraftLinks = nextDrafts;
}

function getPreferredTheme() {
  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === "light" || saved === "dark") {
      return saved;
    }
  } catch {
    // ignore storage access restrictions
  }
  return "dark";
}

function applyTheme(theme) {
  const value = theme === "dark" ? "dark" : "light";
  document.body.setAttribute("data-theme", value);
  if (el.themeToggleInput) {
    el.themeToggleInput.checked = value === "dark";
  }
}

function setThemePreference(theme) {
  const next = theme === "dark" ? "dark" : "light";
  try {
    localStorage.setItem(THEME_STORAGE_KEY, next);
  } catch {
    // ignore storage access restrictions
  }
  applyTheme(next);
}

function loadSystemPromptFromStorage() {
  if (!el.systemPromptInput) return;
  if (Array.isArray(state.workspaceProfiles) && state.workspaceProfiles.length > 0) return;
  try {
    const saved = localStorage.getItem(SYSTEM_PROMPT_STORAGE_KEY);
    if (saved !== null) {
      el.systemPromptInput.value = saved;
    }
  } catch {
    // ignore storage access restrictions
  }
}

function persistSystemPrompt() {
  if (!el.systemPromptInput) return;
  if (String(state.activeWorkspaceProfileId || "").trim()) return;
  try {
    localStorage.setItem(SYSTEM_PROMPT_STORAGE_KEY, String(el.systemPromptInput.value || ""));
  } catch {
    // ignore storage access restrictions
  }
}

function activeModelProfileId() {
  return String(state.models?.active_profile_id || "").trim();
}

function findModelProfile(profileId) {
  const id = String(profileId || "").trim();
  if (!id) return null;
  const rows = Array.isArray(state.models?.profiles) ? state.models.profiles : [];
  return rows.find((item) => String(item?.id || "") === id) || null;
}

function findModelProfileByPath(modelPath) {
  const target = normalizePath(modelPath);
  if (!target) return null;
  const rows = Array.isArray(state.models?.profiles) ? state.models.profiles : [];
  return rows.find((item) => normalizePath(item?.model_path || "") === target) || null;
}

function _safeWorkspaceName(rawName) {
  const value = String(rawName || "").trim();
  return value || "Profile";
}

function _normalizeWorkspaceProfile(raw) {
  if (!raw || typeof raw !== "object") return null;
  const id = String(raw.id || "").trim();
  const name = _safeWorkspaceName(raw.name);
  if (!id) return null;
  const generation = raw.generation && typeof raw.generation === "object" ? raw.generation : {};
  return {
    id,
    name,
    model_profile_id: String(raw.model_profile_id || activeModelProfileId() || "").trim(),
    rag_profile_id: String(raw.rag_profile_id || id).trim(),
    system_prompt: String(raw.system_prompt ?? PROFILE_DEFAULTS.system_prompt),
    generation: {
      temperature: Number(generation.temperature ?? PROFILE_DEFAULTS.temperature),
      top_p: Number(generation.top_p ?? PROFILE_DEFAULTS.top_p),
      max_tokens: Number(generation.max_tokens ?? PROFILE_DEFAULTS.max_tokens),
      repeat_penalty: Number(generation.repeat_penalty ?? PROFILE_DEFAULTS.repeat_penalty),
      rag_top_k: Number(generation.rag_top_k ?? PROFILE_DEFAULTS.rag_top_k),
      rag_enabled: generation.rag_enabled === undefined ? !!PROFILE_DEFAULTS.rag_enabled : !!generation.rag_enabled,
      seed: String(generation.seed ?? PROFILE_DEFAULTS.seed).trim(),
    },
    created_at: Number(raw.created_at || Date.now() / 1000),
    updated_at: Number(raw.updated_at || Date.now() / 1000),
  };
}

function _defaultWorkspaceProfile(profileId, profileName) {
  const now = Date.now() / 1000;
  return _normalizeWorkspaceProfile({
    id: String(profileId || "").trim(),
    name: _safeWorkspaceName(profileName),
    model_profile_id: activeModelProfileId(),
    rag_profile_id: String(profileId || "").trim(),
    system_prompt: String(el.systemPromptInput?.value || PROFILE_DEFAULTS.system_prompt),
    generation: {
      temperature: Number(el.temperatureInput?.value || PROFILE_DEFAULTS.temperature),
      top_p: Number(el.topPInput?.value || PROFILE_DEFAULTS.top_p),
      max_tokens: Number(el.maxTokensInput?.value || PROFILE_DEFAULTS.max_tokens),
      repeat_penalty: Number(el.repeatPenaltyInput?.value || PROFILE_DEFAULTS.repeat_penalty),
      rag_top_k: Number(el.ragTopKInput?.value || PROFILE_DEFAULTS.rag_top_k),
      rag_enabled: !!el.ragEnabled?.checked,
      seed: String(el.seedInput?.value || PROFILE_DEFAULTS.seed).trim(),
    },
    created_at: now,
    updated_at: now,
  });
}

function _upsertWorkspaceProfile(profile) {
  const normalized = _normalizeWorkspaceProfile(profile);
  if (!normalized) return;
  const rows = Array.isArray(state.workspaceProfiles) ? state.workspaceProfiles : [];
  const idx = rows.findIndex((row) => String(row.id || "") === String(normalized.id || ""));
  if (idx < 0) {
    state.workspaceProfiles = [...rows, normalized];
    return;
  }
  const next = [...rows];
  next[idx] = normalized;
  state.workspaceProfiles = next;
}

function syncWorkspaceProfilesWithRagProfiles() {
  const ragProfiles = getRagProfilesArray();
  const rows = (Array.isArray(state.workspaceProfiles) ? state.workspaceProfiles : []).map(_normalizeWorkspaceProfile).filter(Boolean);
  const byId = new Map(rows.map((row) => [String(row.id || ""), row]));
  const byRagId = new Map(rows.map((row) => [String(row.rag_profile_id || ""), row]));
  const byName = new Map(rows.map((row) => [String(row.name || "").trim().toLowerCase(), row]));

  const nextRows = [];
  for (const rag of ragProfiles) {
    const ragId = String(rag?.id || "").trim();
    if (!ragId) continue;
    const ragName = _safeWorkspaceName(rag?.name || ragId);
    const match =
      byId.get(ragId) ||
      byRagId.get(ragId) ||
      byName.get(ragName.toLowerCase()) ||
      null;
    const base = match || _defaultWorkspaceProfile(ragId, ragName);
    nextRows.push(
      _normalizeWorkspaceProfile({
        ...base,
        id: ragId,
        name: ragName,
        rag_profile_id: ragId,
      })
    );
  }

  state.workspaceProfiles = nextRows.filter(Boolean);
  const currentActive = String(state.activeWorkspaceProfileId || "").trim();
  const backendActive = activeRagProfileId();
  let nextActive = "";
  if (state.workspaceProfiles.some((row) => String(row.id || "") === backendActive)) {
    nextActive = backendActive;
  } else if (state.workspaceProfiles.some((row) => String(row.id || "") === currentActive)) {
    nextActive = currentActive;
  } else if (state.workspaceProfiles.length > 0) {
    nextActive = String(state.workspaceProfiles[0].id || "");
  }
  state.activeWorkspaceProfileId = nextActive;
  renderWorkspaceProfileSelect();
  _persistWorkspaceProfilesToStorage();
}

function _persistWorkspaceProfilesToStorage() {
  try {
    localStorage.setItem(
      WORKSPACE_PROFILE_STORAGE_KEY,
      JSON.stringify(
        {
          active_profile_id: state.activeWorkspaceProfileId || "",
          profiles: state.workspaceProfiles || [],
        },
        null,
        2
      )
    );
  } catch {
    // ignore storage access restrictions
  }
}

function loadWorkspaceProfilesFromStorage() {
  state.workspaceProfiles = [];
  state.activeWorkspaceProfileId = "";
  try {
    const raw = localStorage.getItem(WORKSPACE_PROFILE_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    const profiles = Array.isArray(parsed?.profiles) ? parsed.profiles : [];
    state.workspaceProfiles = profiles.map(_normalizeWorkspaceProfile).filter(Boolean);
    const activeId = String(parsed?.active_profile_id || "").trim();
    if (state.workspaceProfiles.some((row) => String(row.id) === activeId)) {
      state.activeWorkspaceProfileId = activeId;
    }
  } catch {
    state.workspaceProfiles = [];
    state.activeWorkspaceProfileId = "";
  }
}

function findWorkspaceProfile(profileId) {
  const id = String(profileId || "").trim();
  if (!id) return null;
  const rows = Array.isArray(state.workspaceProfiles) ? state.workspaceProfiles : [];
  return rows.find((item) => String(item.id || "") === id) || null;
}

function renderWorkspaceProfileSelect() {
  if (!el.workspaceProfileSelect) return;
  const rows = Array.isArray(state.workspaceProfiles) ? state.workspaceProfiles : [];
  const selectedId = String(state.activeWorkspaceProfileId || "").trim();

  if (rows.length === 0) {
    el.workspaceProfileSelect.innerHTML = `<option value="">No profiles</option>`;
    el.workspaceProfileSelect.value = "";
    el.workspaceProfileSelect.disabled = true;
    return;
  }

  const options = [];
  for (const row of rows) {
    const id = String(row.id || "");
    const name = String(row.name || id || "profile");
    options.push(`<option value="${escapeAttr(id)}">${escapeHtml(name)}</option>`);
  }
  el.workspaceProfileSelect.innerHTML = options.join("");
  el.workspaceProfileSelect.disabled = false;
  if (selectedId && rows.some((row) => String(row.id) === selectedId)) {
    el.workspaceProfileSelect.value = selectedId;
  } else {
    const firstId = String(rows[0].id || "");
    state.activeWorkspaceProfileId = firstId;
    el.workspaceProfileSelect.value = firstId;
  }
}

function captureWorkspaceProfileSnapshot(profileName, existingProfile = null) {
  const now = Date.now() / 1000;
  const name = _safeWorkspaceName(profileName);
  const profileId = String(existingProfile?.id || existingProfile?.rag_profile_id || "").trim();
  if (!profileId) {
    return null;
  }
  const ragId = String(existingProfile?.rag_profile_id || profileId).trim();
  return {
    id: profileId,
    name,
    model_profile_id: activeModelProfileId(),
    rag_profile_id: ragId || "",
    system_prompt: String(el.systemPromptInput?.value || ""),
    generation: {
      temperature: Number(el.temperatureInput?.value || 0.8),
      top_p: Number(el.topPInput?.value || 0.95),
      max_tokens: Number(el.maxTokensInput?.value || 512),
      repeat_penalty: Number(el.repeatPenaltyInput?.value || 1.05),
      rag_top_k: Number(el.ragTopKInput?.value || 4),
      rag_enabled: !!el.ragEnabled?.checked,
      seed: String(el.seedInput?.value || "").trim(),
    },
    created_at: Number(existingProfile?.created_at || now),
    updated_at: now,
  };
}

function syncActiveWorkspaceProfileFromUi(opts = {}) {
  const options = opts && typeof opts === "object" ? opts : {};
  const profileId = String(state.activeWorkspaceProfileId || "").trim();
  const existing = findWorkspaceProfile(profileId);
  if (!profileId || !existing) return;

  const snapshot = captureWorkspaceProfileSnapshot(existing.name, existing);
  if (!snapshot) return;
  snapshot.model_profile_id = String(activeModelProfileId() || snapshot.model_profile_id || "").trim();
  snapshot.rag_profile_id = String(existing.rag_profile_id || existing.id || snapshot.rag_profile_id || "").trim();

  _upsertWorkspaceProfile(snapshot);
  if (options.render !== false) {
    renderWorkspaceProfileSelect();
  }
  if (options.persist !== false) {
    _persistWorkspaceProfilesToStorage();
  }
}

function _normalizeConversationMessage(raw) {
  if (!raw || typeof raw !== "object") return null;
  const content = String(raw.content || "");
  const roleRaw = String(raw.role || "").trim().toLowerCase();
  const role = roleRaw === "user" || roleRaw === "assistant" || roleRaw === "system" ? roleRaw : "assistant";
  return { role, content };
}

function _normalizeConversationRecord(raw) {
  if (!raw || typeof raw !== "object") return null;
  const id = String(raw.id || "").trim() || _conversationId();
  const name = String(raw.name || "").trim() || "Conversation";
  const createdAt = Number(raw.created_at || Date.now() / 1000);
  const updatedAt = Number(raw.updated_at || createdAt);
  const messagesRaw = Array.isArray(raw.messages) ? raw.messages : [];
  const messages = messagesRaw.map(_normalizeConversationMessage).filter(Boolean);
  return {
    id,
    name,
    messages,
    created_at: Number.isFinite(createdAt) ? createdAt : Date.now() / 1000,
    updated_at: Number.isFinite(updatedAt) ? updatedAt : Date.now() / 1000,
  };
}

function persistConversationsToStorage() {
  try {
    const conversations = (Array.isArray(state.conversations) ? state.conversations : [])
      .map(_normalizeConversationRecord)
      .filter(Boolean);
    const payload = {
      active_conversation_id: String(state.activeConversationId || "").trim(),
      conversations,
    };
    localStorage.setItem(CONVERSATION_STORAGE_KEY, JSON.stringify(payload, null, 2));
  } catch {
    // ignore storage access restrictions
  }
}

function loadConversationsFromStorage() {
  try {
    const raw = localStorage.getItem(CONVERSATION_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    const rows = Array.isArray(parsed?.conversations) ? parsed.conversations : [];
    const normalized = rows.map(_normalizeConversationRecord).filter(Boolean);
    state.conversations = normalized;
    state.activeConversationId = String(parsed?.active_conversation_id || "").trim();
  } catch {
    // ignore parse/storage failures
  }
}

async function applyWorkspaceProfileById(profileId, opts = {}) {
  const options = opts && typeof opts === "object" ? opts : {};
  const silent = !!options.silent;
  const resetChat = !!options.resetChat;
  const normalizeModel = options.normalizeModel !== false;
  const activateRag = options.activateRag !== false;
  const profile = findWorkspaceProfile(profileId);
  if (!profile) {
    if (!silent) {
      alert("Profile not found.");
    }
    return;
  }

  state.activeWorkspaceProfileId = String(profile.id || "");
  renderWorkspaceProfileSelect();
  _persistWorkspaceProfilesToStorage();

  if (el.systemPromptInput) {
    el.systemPromptInput.value = String(profile.system_prompt || "");
    persistSystemPrompt();
  }
  if (el.temperatureInput) el.temperatureInput.value = String(profile.generation.temperature ?? 0.8);
  if (el.topPInput) el.topPInput.value = String(profile.generation.top_p ?? 0.95);
  if (el.maxTokensInput) el.maxTokensInput.value = String(profile.generation.max_tokens ?? 512);
  if (el.repeatPenaltyInput) el.repeatPenaltyInput.value = String(profile.generation.repeat_penalty ?? 1.05);
  if (el.ragTopKInput) el.ragTopKInput.value = String(profile.generation.rag_top_k ?? 4);
  if (el.ragEnabled) el.ragEnabled.checked = !!profile.generation.rag_enabled;
  if (el.seedInput) el.seedInput.value = String(profile.generation.seed || "");
  updateRagContextControl();

  if (resetChat) {
    setActiveConversationMessages([]);
    state.lastRetrieval = [];
    state.lastTokenUsage = null;
  }

  const warnings = [];

  const modelProfileId = String(profile.model_profile_id || "").trim();
  if (normalizeModel && modelProfileId) {
    if (findModelProfile(modelProfileId)) {
      await activateModelProfile(modelProfileId, { suppressExternalAlert: true });
    } else {
      warnings.push(`Model profile missing: ${modelProfileId}`);
    }
  } else if (normalizeModel && !modelProfileId) {
    await resetModelToBase({ silent: true });
  }

  const ragProfileId = String(profile.rag_profile_id || "").trim();
  if (ragProfileId) {
    if (findRagProfile(ragProfileId)) {
      if (el.ragProfileSelect) {
        el.ragProfileSelect.value = ragProfileId;
      }
      state.ragProfileDraftMode = false;
      syncRagProfileEditorFromSelection();
      if (activateRag && ragProfileId !== activeRagProfileId()) {
        const payload = await callJson("/api/rag/profiles/activate", {
          method: "POST",
          body: JSON.stringify({ profile_id: ragProfileId }),
        });
        applyRagProfilesStatus(payload);
      }
    } else {
      warnings.push(`RAG profile missing: ${ragProfileId}`);
    }
  } else {
    if (el.ragProfileSelect) {
      el.ragProfileSelect.value = "";
    }
    state.ragProfileDraftMode = true;
    syncRagProfileEditorFromSelection();
  }

  renderTokenUsageBadge();
  renderModelContextBar();
  await refreshRag();
  await refreshRagFiles();
  updateInspector();

  if (!silent && warnings.length > 0) {
    alert(`Applied profile with warnings:\n${warnings.join("\n")}`);
  }
}

async function createWorkspaceProfile() {
  const rawName = prompt("Profile name:");
  const name = _safeWorkspaceName(rawName);
  if (!String(rawName || "").trim()) return;
  try {
    const payload = await callJson("/api/rag/profiles/create", {
      method: "POST",
      body: JSON.stringify({
        name,
        description: "",
        source_paths: [],
        set_active: true,
      }),
    });
    state.lastResponse = payload;
    state.ragProfileDraftMode = false;
    applyRagProfilesStatus(payload);
    const profileId = String(payload?.active_profile_id || activeRagProfileId() || "").trim();
    if (profileId) {
      const existing = findWorkspaceProfile(profileId) || {
        id: profileId,
        name,
        rag_profile_id: profileId,
      };
      const snapshot = captureWorkspaceProfileSnapshot(name, existing);
      if (snapshot) {
        _upsertWorkspaceProfile(snapshot);
      }
      state.activeWorkspaceProfileId = profileId;
      renderWorkspaceProfileSelect();
      _persistWorkspaceProfilesToStorage();
    }
    await refreshModels(false);
  } catch (err) {
    alert(`Create profile failed: ${err.message || err}`);
  } finally {
    updateInspector();
  }
}

async function saveWorkspaceProfile() {
  const selectedId = String(el.workspaceProfileSelect?.value || "").trim();
  const existing = findWorkspaceProfile(selectedId);
  if (!existing) {
    alert("Select a profile first.");
    return;
  }
  const snapshot = captureWorkspaceProfileSnapshot(existing.name, existing);
  if (!snapshot) {
    alert("Select a valid profile first.");
    return;
  }
  _upsertWorkspaceProfile(snapshot);
  state.activeWorkspaceProfileId = String(existing.id || "");
  renderWorkspaceProfileSelect();
  _persistWorkspaceProfilesToStorage();
  updateInspector();
}

async function deleteWorkspaceProfile() {
  const selectedId = String(el.workspaceProfileSelect?.value || "").trim();
  const ragProfile = findRagProfile(selectedId);
  if (!selectedId || !ragProfile) {
    alert("Select a profile first.");
    return;
  }
  if (!confirm(`Delete profile '${ragProfile.name || ragProfile.id}'?`)) return;

  try {
    const payload = await callJson("/api/rag/profiles/remove", {
      method: "POST",
      body: JSON.stringify({ profile_id: selectedId }),
    });
    state.lastResponse = payload;
    state.ragProfileDraftMode = false;
    const ragStatus = getRagProfilesPayload(payload) || payload?.rag_profiles || null;
    if (ragStatus) {
      applyRagProfilesStatus(ragStatus);
    } else {
      await refreshRagProfiles();
    }
    if (payload?.models) {
      applyModelStatus(payload.models, false);
    }
    if (state.activeWorkspaceProfileId) {
      await applyWorkspaceProfileById(state.activeWorkspaceProfileId, {
        silent: true,
        resetChat: false,
        normalizeModel: true,
        activateRag: false,
      });
    }
  } catch (err) {
    alert(`Delete profile failed: ${err.message || err}`);
  } finally {
    updateInspector();
  }
}

function pretty(obj) {
  try {
    return JSON.stringify(obj, null, 2);
  } catch (err) {
    return String(obj);
  }
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function compactPath(pathValue, maxLen = 110) {
  const raw = String(pathValue || "").trim();
  if (!raw || raw.length <= maxLen) return raw;
  const keep = Math.max(16, Math.floor((maxLen - 3) / 2));
  return `${raw.slice(0, keep)}...${raw.slice(-keep)}`;
}

function roughTokenCount(text) {
  const raw = String(text || "");
  if (!raw.trim()) return 0;
  const words = raw.trim().split(/\s+/).length;
  const byChars = Math.max(1, Math.round(raw.length / 4));
  return Math.max(words, byChars);
}

function estimateContextTokens() {
  let total = 0;
  total += roughTokenCount(el.systemPromptInput?.value || "");
  for (const message of state.messages || []) {
    if (!(message.role === "user" || message.role === "assistant")) continue;
    total += roughTokenCount(message.content || "");
    total += 4;
  }
  const draft = String(el.userInput?.value || "").trim();
  if (draft) {
    total += roughTokenCount(draft);
    total += 4;
  }
  return Math.max(0, total);
}

function hasAvailableRagContext() {
  const files = Array.isArray(state.ragFiles) ? state.ragFiles : [];
  if (files.some((file) => Number(file?.chunk_count || 0) > 0)) {
    return true;
  }
  const indexedChunks = Number(state.ragStatus?.profile_scope?.indexed_chunks || 0);
  return Number.isFinite(indexedChunks) && indexedChunks > 0;
}

function updateRagContextControl() {
  const hasContext = hasAvailableRagContext();
  if (el.ragEnabled) {
    el.ragEnabled.hidden = !hasContext;
    el.ragEnabled.disabled = !hasContext;
  }
  if (el.ragContextLabel) {
    el.ragContextLabel.textContent = hasContext ? "USE RAG CONTEXT" : "NO RAG CONTEXT PROVIDED";
  }
  if (el.ragContextToggleWrap) {
    el.ragContextToggleWrap.classList.toggle("no-rag-context", !hasContext);
  }
}

function renderTokenUsageBadge() {
  if (!el.tokenUsageBadge) return;
  const contextEstimate = estimateContextTokens();
  const usage = state.lastTokenUsage || {};
  const p = Number(usage.prompt_tokens);
  const c = Number(usage.completion_tokens);
  const t = Number(usage.total_tokens);
  const usageKnown = Number.isFinite(t) && t > 0;

  let suffix = "Last: n/a";
  if (usageKnown) {
    const pText = Number.isFinite(p) && p >= 0 ? p : "?";
    const cText = Number.isFinite(c) && c >= 0 ? c : "?";
    suffix = `Last: p${pText} c${cText} t${t}`;
  }
  el.tokenUsageBadge.textContent = `Ctx ~${contextEstimate} tok | ${suffix}`;
}

function selectedRuntimeModelPath() {
  return String(el.modelPathInput?.value || "").trim();
}

async function applyRuntimeModelSelection() {
  state.runtimeModelPath = selectedRuntimeModelPath();
  renderModelContextBar();
  updateInspector();

  const selectedPath = String(state.runtimeModelPath || "").trim();
  const selectedProfile = findModelProfileByPath(selectedPath);
  const selectedProfileId = String(selectedProfile?.id || "").trim();
  const activeProfileId = activeModelProfileId();
  if (!selectedProfileId || selectedProfileId === activeProfileId) {
    return;
  }

  if (state.runtimeModelSwitching) {
    return;
  }
  state.runtimeModelSwitching = true;
  try {
    await activateModelProfile(selectedProfileId, { suppressExternalAlert: true });
  } finally {
    state.runtimeModelSwitching = false;
    updateRuntimeModelSelect();
  }
}

function getActiveModelProfile() {
  const profile = state.models?.active_profile;
  return profile && typeof profile === "object" ? profile : null;
}

function isTrainingAllowed() {
  const active = getActiveModelProfile();
  return !!active && !active.is_base;
}

function updateTrainingControls() {
  const active = getActiveModelProfile();
  const blocked = !active || !!active.is_base;

  if (el.startTrainBtn) {
    el.startTrainBtn.disabled = blocked;
    el.startTrainBtn.classList.toggle("control-faded", blocked);
  }
  if (el.trainGuardNote) {
    if (blocked) {
      el.trainGuardNote.textContent =
        "Training is disabled while Base BitNet is active. Activate a custom model profile in Models first.";
    } else {
      const name = String(active?.name || "custom profile");
      el.trainGuardNote.textContent = `Training target profile: ${name}`;
    }
  }
}

function updateRuntimeModelSelect() {
  if (!el.modelPathInput) return;

  const profiles = Array.isArray(state.models?.profiles) ? state.models.profiles : [];
  const activeProfileId = String(state.models?.active_profile_id || "");
  const activeProfilePath = String(state.models?.active_profile?.model_path || "").trim();
  const running = !!state.serverStatus?.running;
  const loadedPath = String(state.serverStatus?.model_path || "").trim();
  const previousSelection = String(state.runtimeModelPath || selectedRuntimeModelPath()).trim();

  const options = [];
  const knownPaths = new Set();
  for (const profile of profiles) {
    const path = String(profile?.model_path || "").trim();
    if (!path) continue;
    knownPaths.add(normalizePath(path));
  }

  if (running && loadedPath && !knownPaths.has(normalizePath(loadedPath))) {
    options.push({
      value: loadedPath,
      label: `Loaded (unregistered): ${compactPath(loadedPath, 72)}`,
      disabled: false,
    });
  }

  for (const profile of profiles) {
    const profileId = String(profile?.id || "");
    const path = String(profile?.model_path || "").trim();
    if (!path) continue;
    const exists = !!profile?.exists;
    const isActive = profileId && profileId === activeProfileId;
    const isLoaded = running && loadedPath && normalizePath(path) === normalizePath(loadedPath);
    const tags = [];
    if (profile?.is_base) tags.push("base");
    if (isActive) tags.push("active");
    if (isLoaded) tags.push("loaded");
    if (!exists) tags.push("missing");
    const suffix = tags.length ? ` [${tags.join(", ")}]` : "";
    options.push({
      value: path,
      label: `${String(profile?.name || profileId || "profile")}${suffix}`,
      disabled: !exists,
    });
  }

  if (options.length === 0) {
    el.modelPathInput.innerHTML = `<option value="">No runtime model profiles available</option>`;
    el.modelPathInput.disabled = true;
    state.runtimeModelPath = "";
    if (el.runtimeModelPathNote) {
      el.runtimeModelPathNote.textContent = "No model profiles yet. Add one in the Models tab.";
    }
    return;
  }

  el.modelPathInput.disabled = false;

  const preferredCandidates = [previousSelection, activeProfilePath, loadedPath].filter((v) => String(v || "").trim());
  let selectedIndex = -1;
  for (const candidate of preferredCandidates) {
    const key = normalizePath(candidate);
    const candidateIndex = options.findIndex((item) => normalizePath(item.value) === key && !item.disabled);
    if (candidateIndex >= 0) {
      selectedIndex = candidateIndex;
      break;
    }
  }
  if (selectedIndex < 0) {
    selectedIndex = options.findIndex((item) => !item.disabled);
  }
  if (selectedIndex < 0) {
    selectedIndex = 0;
  }

  el.modelPathInput.innerHTML = options
    .map((item, index) => {
      const selected = index === selectedIndex ? " selected" : "";
      const disabled = item.disabled ? " disabled" : "";
      return `<option value="${escapeAttr(item.value)}"${selected}${disabled}>${escapeHtml(item.label)}</option>`;
    })
    .join("");

  state.runtimeModelPath = selectedRuntimeModelPath();
  renderChatRuntimeLabel();

  if (el.runtimeModelPathNote) {
    const selectedPath = state.runtimeModelPath;
    if (running && loadedPath && selectedPath && normalizePath(selectedPath) !== normalizePath(loadedPath)) {
      el.runtimeModelPathNote.textContent = "Server is running with a different model; your selection will apply after restart.";
    } else if (running && loadedPath) {
      el.runtimeModelPathNote.textContent = "Server is running with the selected model.";
    } else {
      el.runtimeModelPathNote.textContent = "Choose which registered model profile to run before starting the server.";
    }
  }
}

function renderChatRuntimeLabel() {
  if (!el.chatRuntimeLabel) return;
  const running = !!state.serverStatus?.running;
  const loadedPathRaw = String(state.serverStatus?.model_path || "").trim();
  const loadedPath = normalizePath(loadedPathRaw);
  const profiles = Array.isArray(state.models?.profiles) ? state.models.profiles : [];

  if (running && loadedPathRaw) {
    const loadedProfile = profiles.find((profile) => normalizePath(profile?.model_path || "") === loadedPath) || null;
    if (loadedProfile?.name) {
      el.chatRuntimeLabel.textContent = String(loadedProfile.name);
      return;
    }
    el.chatRuntimeLabel.textContent = compactPath(loadedPathRaw, 52) || "Runtime model";
    return;
  }

  const activeProfileName = String(state.models?.active_profile?.name || "").trim();
  if (activeProfileName) {
    el.chatRuntimeLabel.textContent = activeProfileName;
    return;
  }

  const selectedPath = selectedRuntimeModelPath();
  if (selectedPath) {
    el.chatRuntimeLabel.textContent = compactPath(selectedPath, 52);
    return;
  }

  el.chatRuntimeLabel.textContent = "...";
}

function renderModelContextBar() {
  renderChatRuntimeLabel();
  if (!el.modelContextBar) return;

  const running = !!state.serverStatus?.running;
  const selectedPath = selectedRuntimeModelPath() || "not selected";
  const loadedPathRaw = String(state.serverStatus?.model_path || "").trim();
  const loadedPath = running ? (loadedPathRaw || "running (model path unavailable)") : "server stopped";

  const activeProfile = state.models?.active_profile || null;
  const profileName = activeProfile?.name || "unknown";
  const profilePath = String(activeProfile?.model_path || "").trim();
  const profileExists = activeProfile ? !!activeProfile.exists : false;
  const ragChunks = Number(state.ragStatus?.total_chunks || 0);
  const activeRag = state.ragProfiles?.active_profile || null;
  const activeRagName = activeRag?.name || "none";
  const activeRagSources = Number(activeRag?.source_count || 0);
  const trainBase = String(el.trainBaseModelInput?.value || "").trim() || "not set";
  const managedState = running
    ? (state.serverStatus?.managed ? "managed" : state.serverStatus?.external ? "external" : "unknown")
    : "stopped";

  let syncLabel = running ? "no active profile" : "not running";
  if (running) {
    if (profilePath && loadedPathRaw) {
      syncLabel = normalizePath(profilePath) === normalizePath(loadedPathRaw) ? "synced" : "profile mismatch";
    } else {
      syncLabel = "unverified";
    }
  }

  const lines = [
    `<div class="model-context-line"><span class="model-context-key">Selected runtime model:</span>${escapeHtml(
      compactPath(selectedPath)
    )}</div>`,
    `<div class="model-context-line"><span class="model-context-key">Loaded by llama-server:</span>${escapeHtml(
      compactPath(loadedPath)
    )} (${escapeHtml(managedState)})</div>`,
    `<div class="model-context-line"><span class="model-context-key">Active profile:</span>${escapeHtml(profileName)} (${escapeHtml(
      profileExists ? "ready" : "missing"
    )}, ${escapeHtml(syncLabel)})</div>`,
    `<div class="model-context-line"><span class="model-context-key">Training base:</span>${escapeHtml(
      compactPath(trainBase)
    )} (BitNet continued training)</div>`,
    `<div class="model-context-line"><span class="model-context-key">Active RAG profile:</span>${escapeHtml(
      `${activeRagName}${activeRagSources ? ` (${activeRagSources} source${activeRagSources === 1 ? "" : "s"})` : ""}`
    )}</div>`,
    `<div class="model-context-line"><span class="model-context-key">RAG chunks:</span>${ragChunks}</div>`,
  ];

  el.modelContextBar.innerHTML = lines.join("");
}

function updateInspector() {
  renderTokenUsageBadge();
  el.inspector.textContent = pretty({
    server: state.serverStatus,
    models: state.models
      ? {
          active_profile_id: state.models.active_profile_id,
          active_profile: state.models.active_profile,
          profile_count: Array.isArray(state.models.profiles) ? state.models.profiles.length : 0,
        }
      : null,
    rag: state.ragStatus,
    rag_profiles: state.ragProfiles
      ? {
          active_profile_id: state.ragProfiles.active_profile_id,
          profile_count: Array.isArray(state.ragProfiles.profiles) ? state.ragProfiles.profiles.length : 0,
        }
      : null,
    rag_files_count: Array.isArray(state.ragFiles) ? state.ragFiles.length : 0,
    selected_rag_path: state.selectedRagPath,
    profiles: {
      active_profile_id: state.activeWorkspaceProfileId || null,
      profile_count: Array.isArray(state.workspaceProfiles) ? state.workspaceProfiles.length : 0,
    },
    conversations: {
      active_conversation_id: state.activeConversationId || null,
      conversation_count: Array.isArray(state.conversations) ? state.conversations.length : 0,
      active_message_count: Array.isArray(state.messages) ? state.messages.length : 0,
    },
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

function setActiveSettingsSubtab(subtabName) {
  const target = String(subtabName || "").trim() || "runtime";
  state.settingsSubtab = target;
  for (const button of el.settingsSubtabButtons) {
    button.classList.toggle("active", button.dataset.settingsSubtab === target);
  }
  for (const panel of el.settingsSubtabPanels) {
    panel.classList.toggle("active", panel.dataset.settingsSubpanel === target);
  }
}

function setActiveRagSubtab(subtabName) {
  const target = String(subtabName || "").trim() || "files";
  state.ragSubtab = target;
  for (const button of el.ragSubtabButtons) {
    button.classList.toggle("active", button.dataset.ragSubtab === target);
  }
  for (const panel of el.ragSubtabPanels) {
    panel.classList.toggle("active", panel.dataset.ragSubpanel === target);
  }
}

function _conversationId() {
  return `conv-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function _nextConversationName() {
  const count = Array.isArray(state.conversations) ? state.conversations.length : 0;
  return `Conversation ${count + 1}`;
}

function _createConversationRecord(name, messages = [], conversationId = "") {
  const safeName = String(name || "").trim() || _nextConversationName();
  const safeMessages = Array.isArray(messages) ? messages : [];
  const now = Date.now() / 1000;
  return {
    id: String(conversationId || _conversationId()),
    name: safeName,
    messages: safeMessages,
    created_at: now,
    updated_at: now,
  };
}

function findConversationById(conversationId) {
  const id = String(conversationId || "").trim();
  if (!id) return null;
  return (state.conversations || []).find((row) => String(row?.id || "") === id) || null;
}

function getActiveConversation() {
  return findConversationById(state.activeConversationId);
}

function ensureConversationState() {
  let dirty = false;
  if (!Array.isArray(state.conversations)) {
    state.conversations = [];
    dirty = true;
  }
  if (state.conversations.length === 0) {
    const fresh = _createConversationRecord("Conversation 1", []);
    state.conversations.push(fresh);
    state.activeConversationId = String(fresh.id || "");
    state.messages = fresh.messages;
    persistConversationsToStorage();
    return;
  }

  const active = getActiveConversation();
  if (!active) {
    const first = state.conversations[0];
    state.activeConversationId = String(first?.id || "");
    state.messages = Array.isArray(first?.messages) ? first.messages : [];
    dirty = true;
    persistConversationsToStorage();
    return;
  }
  if (!Array.isArray(active.messages)) {
    active.messages = [];
    dirty = true;
  }
  state.messages = active.messages;
  if (dirty) {
    persistConversationsToStorage();
  }
}

function renderConversationList() {
  if (!el.conversationList) return;
  ensureConversationState();
  const activeId = String(state.activeConversationId || "");
  const rows = (state.conversations || [])
    .map((conversation) => {
      const id = String(conversation.id || "");
      const active = id === activeId;
      const name = String(conversation.name || id || "Conversation");
      return `
        <div class="conversation-item ${active ? "is-active" : ""}" data-conversation-id="${escapeAttr(id)}">
          <button class="conversation-select" type="button" data-conversation-action="switch" data-conversation-id="${escapeAttr(id)}">
            <span class="conversation-name">${escapeHtml(name)}</span>
          </button>
          <span class="conversation-actions">
            <button class="conversation-action-btn icon-edit" type="button" data-conversation-action="rename" data-conversation-id="${escapeAttr(id)}" aria-label="Rename conversation"></button>
            <button class="conversation-action-btn icon-delete" type="button" data-conversation-action="delete" data-conversation-id="${escapeAttr(id)}" aria-label="Delete conversation"></button>
          </span>
        </div>
      `;
    })
    .join("");
  el.conversationList.innerHTML = rows;
}

function setActiveConversationMessages(nextMessages) {
  ensureConversationState();
  const active = getActiveConversation();
  const safe = Array.isArray(nextMessages) ? nextMessages : [];
  if (active) {
    active.messages = safe;
    active.updated_at = Date.now() / 1000;
  }
  state.messages = safe;
  persistConversationsToStorage();
  renderConversationList();
  renderMessages();
}

function switchConversation(conversationId) {
  if (state.streamAbort) return;
  const target = findConversationById(conversationId);
  if (!target) return;
  setActiveTab("chat");
  state.activeConversationId = String(target.id || "");
  state.messages = Array.isArray(target.messages) ? target.messages : [];
  state.lastRetrieval = [];
  state.lastTokenUsage = null;
  persistConversationsToStorage();
  renderConversationList();
  renderMessages();
  updateInspector();
}

function createConversation() {
  if (state.streamAbort) return;
  ensureConversationState();
  const fresh = _createConversationRecord(_nextConversationName(), []);
  state.conversations = [...(state.conversations || []), fresh];
  state.activeConversationId = String(fresh.id || "");
  state.messages = fresh.messages;
  state.lastRetrieval = [];
  state.lastTokenUsage = null;
  persistConversationsToStorage();
  renderConversationList();
  renderMessages();
  updateInspector();
}

function renameConversation(conversationId) {
  const conversation = findConversationById(conversationId);
  if (!conversation) return;
  const rawName = prompt("Conversation name:", String(conversation.name || ""));
  if (rawName === null) return;
  const nextName = String(rawName || "").trim();
  if (!nextName) return;
  conversation.name = nextName;
  conversation.updated_at = Date.now() / 1000;
  persistConversationsToStorage();
  renderConversationList();
  updateInspector();
}

function deleteConversation(conversationId) {
  if (state.streamAbort) return;
  const conversation = findConversationById(conversationId);
  if (!conversation) return;
  if (!confirm(`Delete conversation '${conversation.name || conversation.id}'?`)) return;

  state.conversations = (state.conversations || []).filter(
    (row) => String(row?.id || "") !== String(conversation.id || "")
  );
  if (state.conversations.length === 0) {
    const fallback = _createConversationRecord("Conversation 1", []);
    state.conversations = [fallback];
  }

  if (!findConversationById(state.activeConversationId)) {
    state.activeConversationId = String(state.conversations[0]?.id || "");
  }

  const active = getActiveConversation();
  state.messages = Array.isArray(active?.messages) ? active.messages : [];
  state.lastRetrieval = [];
  state.lastTokenUsage = null;
  persistConversationsToStorage();
  renderConversationList();
  renderMessages();
  updateInspector();
}

function renderMessages() {
  ensureConversationState();
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
  renderTokenUsageBadge();
}

function setChatStreamingState(isStreaming) {
  const active = !!isStreaming;
  if (el.sendBtn) {
    el.sendBtn.disabled = active;
    el.sendBtn.hidden = active;
  }
  if (el.stopBtn) {
    el.stopBtn.hidden = !active;
    el.stopBtn.disabled = !active;
  }
}

function autoResizeComposerInput() {
  if (!el.userInput) return;
  el.userInput.style.height = "0px";
  const minHeight = 38;
  const maxHeight = 136;
  const next = Math.max(minHeight, Math.min(el.userInput.scrollHeight, maxHeight));
  el.userInput.style.height = `${next}px`;
}

function pushMessage(role, content) {
  ensureConversationState();
  state.messages.push({ role, content });
  const active = getActiveConversation();
  if (active) {
    active.updated_at = Date.now() / 1000;
  }
  persistConversationsToStorage();
  renderMessages();
  renderConversationList();
  return state.messages.length - 1;
}

async function callJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const raw = await response.text();
  let payload = {};
  try {
    payload = raw ? JSON.parse(raw) : {};
  } catch {
    payload = { raw };
  }
  if (!response.ok) {
    const detail = payload?.detail || payload?.error || raw || JSON.stringify(payload);
    throw new Error(detail);
  }
  return payload;
}

async function refreshHealth() {
  try {
    const health = await callJson("/api/health");
    const serverStatus = await callJson("/api/server/status");
    state.serverStatus = serverStatus;
    if (!String(state.runtimeModelPath || "").trim() && serverStatus?.running && serverStatus?.model_path) {
      state.runtimeModelPath = String(serverStatus.model_path).trim();
    }
    updateRuntimeModelSelect();
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
    updateRuntimeModelSelect();
    if (!state.serverAction) {
      syncServerButtons(false);
    }
    el.healthBadge.textContent = "Web UI error";
    el.healthBadge.className = "badge bad";
    el.llamaBadge.textContent = String(err.message || err);
    el.llamaBadge.className = "badge bad";
  } finally {
    renderModelContextBar();
    updateInspector();
  }
}

function renderRagProfileSourcesTable() {
  if (!el.ragProfileSourcesTableBody) return;
  const selectedId = String(el.ragProfileSelect?.value || "").trim();
  const profile = findRagProfile(selectedId);
  if (!profile) {
    el.ragProfileSourcesTableBody.innerHTML = `
      <tr>
        <td colspan="3">Select or create a RAG profile.</td>
      </tr>
    `;
    return;
  }

  const sourcePaths = Array.isArray(profile.source_paths) ? profile.source_paths : [];
  if (sourcePaths.length === 0) {
    el.ragProfileSourcesTableBody.innerHTML = `
      <tr>
        <td colspan="3">No sources mapped to this profile yet.</td>
      </tr>
    `;
    return;
  }

  el.ragProfileSourcesTableBody.innerHTML = sourcePaths
    .map((path) => {
      const chunks = chunkCountForSource(path);
      return `
        <tr data-profile-source-path="${escapeAttr(path)}">
          <td class="path-cell">${escapeHtml(path)}</td>
          <td>${chunks}</td>
          <td>
            <div class="mini-actions">
              <button class="btn mini subtle icon-delete" data-rag-profile-action="remove-source" data-path="${escapeAttr(path)}">Remove</button>
            </div>
          </td>
        </tr>
      `;
    })
    .join("");
}

function hasSelectedRagProfile() {
  const selectedId = selectedRagProfileId();
  return !!findRagProfile(selectedId);
}

function updateRagProfileControls() {
  const hasProfiles = getRagProfilesArray().length > 0;
  const selectedExisting = hasSelectedRagProfile();
  const inCreateMode = state.ragProfileDraftMode || !selectedExisting;

  if (el.ragProfileSelect) {
    el.ragProfileSelect.disabled = !hasProfiles;
    el.ragProfileSelect.classList.toggle("control-faded", !hasProfiles);
  }
  if (el.activateRagProfileBtn) {
    el.activateRagProfileBtn.disabled = !selectedExisting;
    el.activateRagProfileBtn.classList.toggle("control-faded", !selectedExisting);
  }
  if (el.removeRagProfileBtn) {
    el.removeRagProfileBtn.disabled = !selectedExisting;
    el.removeRagProfileBtn.classList.toggle("control-faded", !selectedExisting);
  }
  if (el.addSelectedToProfileBtn) {
    el.addSelectedToProfileBtn.disabled = !selectedExisting;
    el.addSelectedToProfileBtn.classList.toggle("control-faded", !selectedExisting);
  }
  if (el.removeSelectedFromProfileBtn) {
    el.removeSelectedFromProfileBtn.disabled = !selectedExisting;
    el.removeSelectedFromProfileBtn.classList.toggle("control-faded", !selectedExisting);
  }
  if (el.saveRagProfileBtn) {
    el.saveRagProfileBtn.textContent = inCreateMode ? "Create Profile" : "Save Profile";
  }
}

function syncRagProfileEditorFromSelection() {
  const selectedId = String(el.ragProfileSelect?.value || "").trim();
  const profile = findRagProfile(selectedId);
  const hasProfile = !!profile;
  state.ragProfileDraftMode = !hasProfile;

  if (hasProfile) {
    if (el.ragProfileNameInput) {
      el.ragProfileNameInput.value = profile?.name || "";
    }
    if (el.ragProfileDescriptionInput) {
      el.ragProfileDescriptionInput.value = profile?.description || "";
    }
  }
  renderRagProfileSourcesTable();
  updateRagProfileControls();
}

function renderRagProfileSelect() {
  if (!el.ragProfileSelect) return;
  const profiles = getRagProfilesArray();
  const activeId = activeRagProfileId();
  const current = String(el.ragProfileSelect.value || "").trim();
  const inDraftMode = !!state.ragProfileDraftMode;
  const candidates = inDraftMode ? [current] : [current, activeId];
  let selectedId = "";
  for (const candidate of candidates) {
    if (!candidate) continue;
    if (profiles.some((item) => String(item.id || "") === candidate)) {
      selectedId = candidate;
      break;
    }
  }
  if (!selectedId && profiles.length > 0 && !inDraftMode) {
    selectedId = String(profiles[0].id || "");
  }

  if (profiles.length === 0) {
    el.ragProfileSelect.innerHTML = `<option value="">No RAG profiles</option>`;
    el.ragProfileSelect.disabled = true;
    state.ragProfileDraftMode = true;
    syncRagProfileEditorFromSelection();
    return;
  }

  el.ragProfileSelect.disabled = false;
  const options = [];
  options.push(`<option value="">New profile draft</option>`);
  for (const profile of profiles) {
    const id = String(profile.id || "");
    const isActive = id === activeId;
    const suffix = isActive ? " [active]" : "";
    options.push(`<option value="${escapeAttr(id)}">${escapeHtml(String(profile.name || id) + suffix)}</option>`);
  }
  el.ragProfileSelect.innerHTML = options.join("");
  el.ragProfileSelect.value = selectedId;
  syncRagProfileEditorFromSelection();
}

function applyRagProfilesStatus(statusPayload) {
  const status =
    getRagProfilesPayload(statusPayload) ||
    (statusPayload && typeof statusPayload === "object" ? statusPayload : null);
  state.ragProfiles = status || null;
  const hasProfiles = getRagProfilesArray().length > 0;
  if (!hasProfiles) {
    state.ragProfileDraftMode = true;
  }
  const prevWorkspaceProfileId = String(state.activeWorkspaceProfileId || "").trim();
  syncWorkspaceProfilesWithRagProfiles();
  const nextWorkspaceProfileId = String(state.activeWorkspaceProfileId || "").trim();
  const shouldApplyProfileSwitch = !!nextWorkspaceProfileId && nextWorkspaceProfileId !== prevWorkspaceProfileId;
  renderRagProfileSelect();
  renderRagProfileSourcesTable();
  renderModelsTable();

  if (el.ragProfilesStatus) {
    const profiles = getRagProfilesArray();
    el.ragProfilesStatus.textContent = pretty({
      active_profile_id: state.ragProfiles?.active_profile_id || null,
      active_profile: state.ragProfiles?.active_profile || null,
      active_profile_code: state.ragProfiles?.active_profile?.code || null,
      active_profile_upload_dir: state.ragProfiles?.active_profile?.upload_dir || null,
      profiles: profiles.map((item) => ({
        id: item.id,
        name: item.name,
        code: item.code || null,
        upload_dir: item.upload_dir || null,
        source_count: item.source_count,
        indexed_chunks: item.indexed_chunks,
        linked_model_count: item.linked_model_count,
      })),
    });
  }

  renderModelContextBar();
  updateRagProfileControls();
  updateTrainingControls();
  if (shouldApplyProfileSwitch) {
    void applyWorkspaceProfileById(nextWorkspaceProfileId, {
      silent: true,
      resetChat: false,
      normalizeModel: true,
      activateRag: false,
    }).catch((err) => {
      console.warn("Profile sync apply failed:", err);
    });
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
    updateRagContextControl();
    renderModelContextBar();
    updateInspector();
  }
}

async function refreshRagProfiles() {
  try {
    const status = await callJson("/api/rag/profiles");
    applyRagProfilesStatus(status);
  } catch (err) {
    if (el.ragProfilesStatus) {
      el.ragProfilesStatus.textContent = `RAG profile status error: ${err.message || err}`;
    }
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
    renderModelContextBar();
    updateInspector();
  }
}

function resolvedModelPathFromForm() {
  const dir = String(el.modelProfileDirInput?.value || "").trim();
  const name = String(el.modelProfileNameInput?.value || "").trim() || "model";
  return joinPath(dir, profileFilename(name));
}

function updateResolvedModelPathPreview() {
  if (!el.modelProfileComputedPath) return;
  el.modelProfileComputedPath.value = resolvedModelPathFromForm();
}

async function pickModelDirectory() {
  if (!el.browseModelDirBtn) return;
  const startPath =
    String(el.modelProfileDirInput?.value || "").trim() ||
    dirnamePath(state.serverStatus?.model_path || "") ||
    "";
  const previousText = el.browseModelDirBtn.textContent;
  el.browseModelDirBtn.disabled = true;
  el.browseModelDirBtn.textContent = "Opening...";
  try {
    const payload = await callJson("/api/fs/pick-dir", {
      method: "POST",
      body: JSON.stringify({ start_path: startPath }),
    });
    if (payload?.cancelled) {
      return;
    }
    const selected = String(payload?.selected_path || "").trim();
    if (!selected) {
      return;
    }
    if (el.modelProfileDirInput) {
      el.modelProfileDirInput.value = selected;
    }
    updateResolvedModelPathPreview();
  } catch (err) {
    alert(`Directory picker failed: ${err.message || err}`);
  } finally {
    el.browseModelDirBtn.disabled = false;
    el.browseModelDirBtn.textContent = previousText || "Browse";
  }
}

function applyModelStatus(status, syncRuntimeModelPath = false) {
  const previousActiveModelId = activeModelProfileId();
  const modelStatus = getModelsPayload(status);
  state.models = modelStatus || null;
  reconcileModelRagDraftLinks(modelStatus);

  const ragProfilePayload = getRagProfilesPayload(status);
  if (ragProfilePayload) {
    applyRagProfilesStatus(ragProfilePayload);
  }
  const nextActiveModelId = String(modelStatus?.active_profile_id || "").trim();
  if (nextActiveModelId && nextActiveModelId !== previousActiveModelId) {
    syncActiveWorkspaceProfileFromUi({ render: false, persist: true });
  }

  if (syncRuntimeModelPath) {
    const activePath = String(modelStatus?.active_profile?.model_path || "").trim();
    if (activePath) {
      state.runtimeModelPath = activePath;
    }
  }
  updateRuntimeModelSelect();

  const active = modelStatus?.active_profile || null;
  const rows = Array.isArray(modelStatus?.profiles) ? modelStatus.profiles.length : 0;
  if (el.modelsStatus) {
    el.modelsStatus.textContent = pretty({
      active_profile_id: modelStatus?.active_profile_id || null,
      active_profile: active
        ? {
            id: active.id,
            name: active.name,
            model_path: active.model_path,
            exists: !!active.exists,
            is_base: !!active.is_base,
          }
        : null,
      profiles: rows,
    });
  }
  renderModelsTable();
  renderModelContextBar();
  updateTrainingControls();
}

function renderModelsTable() {
  if (!el.modelsTableBody) return;
  const profiles = Array.isArray(state.models?.profiles) ? state.models.profiles : [];
  const activeId = String(state.models?.active_profile_id || "");
  const ragProfiles = getRagProfilesArray();

  if (profiles.length === 0) {
    el.modelsTableBody.innerHTML = `
      <tr>
        <td colspan="6">No model profiles registered.</td>
      </tr>
    `;
    return;
  }

  const rows = profiles
    .map((profile) => {
      const profileId = String(profile.id || "");
      const isActive = profileId === activeId;
      const exists = !!profile.exists;
      const typeLabel = profile.is_base ? "Base" : "Custom";
      const statusLabel = isActive ? "active" : (exists ? "ready" : "pending");
      const statusBadge = statusPill(statusLabel, statusTone(statusLabel));
      const existsBadge = statusPill(exists ? "path ready" : "awaiting file", exists ? "good" : "warn");
      const backendLinkedRagId = String(profile.rag_profile_id || "");
      const draftLinkedRagId = hasOwn(state.modelRagDraftLinks || {}, profileId)
        ? String(state.modelRagDraftLinks[profileId] || "")
        : null;
      const linkedRagId = draftLinkedRagId !== null ? draftLinkedRagId : backendLinkedRagId;
      const linkDirty = linkedRagId !== backendLinkedRagId;
      const ragOptions = [
        `<option value=""${linkedRagId ? "" : " selected"}>No linked profile</option>`,
        ...ragProfiles.map((rag) => {
          const ragId = String(rag.id || "");
          const selected = ragId && ragId === linkedRagId ? " selected" : "";
          return `<option value="${escapeAttr(ragId)}"${selected}>${escapeHtml(
            String(rag.name || ragId)
          )}</option>`;
        }),
      ].join("");

      const activateDisabled = isActive || !exists;
      const removeDisabled = !!profile.is_base;
      return `
        <tr data-profile-id="${escapeAttr(profileId)}">
          <td>${escapeHtml(profile.name || profileId)}</td>
          <td>${escapeHtml(typeLabel)}</td>
          <td>${statusBadge} ${existsBadge}</td>
          <td class="path-cell">${escapeHtml(profile.model_path || "")}</td>
          <td>
            <div class="mini-actions">
              <select class="model-rag-select" data-profile-id="${escapeAttr(profileId)}">
                ${ragOptions}
              </select>
              <button class="btn mini icon-save ${linkDirty ? "accent" : "subtle"}" data-model-action="link-rag" data-profile-id="${escapeAttr(profileId)}">${
        linkDirty ? "Save*" : "Save Link"
      }</button>
            </div>
          </td>
          <td>
            <div class="mini-actions">
              <button class="btn mini subtle icon-open" data-model-action="activate" data-profile-id="${escapeAttr(profileId)}" ${
        activateDisabled ? "disabled" : ""
      }>Activate</button>
              <button class="btn mini subtle icon-delete" data-model-action="remove" data-profile-id="${escapeAttr(profileId)}" ${
        removeDisabled ? "disabled" : ""
      }>Remove</button>
            </div>
          </td>
        </tr>
      `;
    })
    .join("");

  el.modelsTableBody.innerHTML = rows;
}

async function refreshModels(syncRuntimeModelPath = false) {
  try {
    const status = await callJson("/api/models");
    applyModelStatus(status, syncRuntimeModelPath);
  } catch (err) {
    if (el.modelsStatus) {
      el.modelsStatus.textContent = `Model status error: ${err.message || err}`;
    }
    if (el.modelsTableBody) {
      el.modelsTableBody.innerHTML = `
        <tr>
          <td colspan="6">Unable to load model profiles.</td>
        </tr>
      `;
    }
  } finally {
    updateInspector();
  }
}

async function maybeRestartServerForModel(newModelPath, opts = {}) {
  const options = opts && typeof opts === "object" ? opts : {};
  const suppressExternalAlert = !!options.suppressExternalAlert;
  const running = !!state.serverStatus?.running;
  if (!running) return;

  const currentPath = normalizePath(state.serverStatus?.model_path || "");
  const nextPath = normalizePath(newModelPath || "");
  if (!nextPath || (currentPath && currentPath === nextPath)) {
    return;
  }

  await stopServer();
  const stillRunning = !!state.serverStatus?.running;
  const stillManaged = !!state.serverStatus?.managed;
  const stillExternal = !!state.serverStatus?.external;
  const stillPath = normalizePath(state.serverStatus?.model_path || "");

  if (stillRunning && stillExternal && !stillManaged && stillPath !== nextPath) {
    const message =
      "Model switch could not be applied automatically because llama-server is running as an external unmanaged process. Stop that process, then start the server again.";
    if (suppressExternalAlert) {
      if (el.runtimeModelPathNote) {
        el.runtimeModelPathNote.textContent =
          "External unmanaged llama-server is running with a different model; stop it to apply profile model changes.";
      }
      console.warn(message);
    } else {
      alert(message);
    }
    return;
  }

  if (stillRunning && stillPath === nextPath) {
    return;
  }

  await startServer();
}

async function registerModelProfile() {
  const modelDir = String(el.modelProfileDirInput?.value || "").trim();
  const name = String(el.modelProfileNameInput?.value || "").trim() || "model";
  const initFromBase = !!el.modelProfileInitFromBaseInput?.checked;
  if (!modelDir) {
    alert("Select a model directory first.");
    return;
  }
  updateResolvedModelPathPreview();
  const resolvedPath = resolvedModelPathFromForm();
  const beforeCount = Array.isArray(state.models?.profiles) ? state.models.profiles.length : 0;
  try {
    const status = await callJson("/api/models/register", {
      method: "POST",
      body: JSON.stringify({
        name,
        model_dir: modelDir,
        initialize_from_base: initFromBase,
        set_active: !!el.modelProfileSetActiveInput.checked,
      }),
    });
    state.lastResponse = status;
    applyModelStatus(status, true);
    const normalizedPath = normalizePath(resolvedPath);
    const savedProfile = (status?.profiles || []).find(
      (row) => normalizePath(row?.model_path || "") === normalizedPath
    );
    const savedExists = !!savedProfile?.exists;
    const savedId = String(savedProfile?.id || "");
    const isActive = savedId && String(status?.active_profile_id || "") === savedId;
    el.modelProfileNameInput.value = "";
    if (el.modelProfileSetActiveInput.checked && savedExists) {
      await maybeRestartServerForModel(status?.active_profile?.model_path || "", {
        suppressExternalAlert: true,
      });
    }
    const afterCount = Array.isArray(status?.profiles) ? status.profiles.length : 0;
    if (!savedExists) {
      alert(
        `Profile saved as pending.\nThe model file is not on disk yet, so it cannot be activated until created:\n${resolvedPath}`
      );
    } else if (afterCount <= beforeCount) {
      alert("That model path is already registered. Existing profile reused.");
    } else if (el.modelProfileSetActiveInput.checked && !isActive) {
      alert("Profile saved, but it was not set active.");
    } else if (initFromBase) {
      alert(`Profile saved and ready.\nBase model copied to:\n${resolvedPath}`);
    } else {
      alert(`Model profile saved: ${name}`);
    }
  } catch (err) {
    alert(`Add model profile failed: ${err.message || err}`);
  } finally {
    updateInspector();
  }
}

async function activateModelProfile(profileId, opts = {}) {
  const options = opts && typeof opts === "object" ? opts : {};
  try {
    const status = await callJson("/api/models/activate", {
      method: "POST",
      body: JSON.stringify({ profile_id: profileId }),
    });
    state.lastResponse = status;
    applyModelStatus(status, true);
    const modelStatus = getModelsPayload(status) || {};
    await maybeRestartServerForModel(modelStatus?.active_profile?.model_path || "", {
      suppressExternalAlert: !!options.suppressExternalAlert,
    });
  } catch (err) {
    alert(`Activate profile failed: ${err.message || err}`);
  } finally {
    updateInspector();
  }
}

async function removeModelProfile(profileId) {
  const profile = (state.models?.profiles || []).find((row) => String(row.id) === String(profileId));
  if (!profile) return;
  if (!confirm(`Remove model profile '${profile.name || profile.id}'?`)) return;

  try {
    const status = await callJson("/api/models/remove", {
      method: "POST",
      body: JSON.stringify({ profile_id: profileId }),
    });
    state.lastResponse = status;
    applyModelStatus(status, true);
  } catch (err) {
    alert(`Remove profile failed: ${err.message || err}`);
  } finally {
    updateInspector();
  }
}

async function resetModelToBase(opts = {}) {
  const options = opts && typeof opts === "object" ? opts : {};
  const silent = !!options.silent;
  try {
    const status = await callJson("/api/models/reset-base", { method: "POST" });
    state.lastResponse = status;
    applyModelStatus(status, true);
    const modelStatus = getModelsPayload(status) || {};
    await maybeRestartServerForModel(modelStatus?.active_profile?.model_path || "", {
      suppressExternalAlert: true,
    });
  } catch (err) {
    if (!silent) {
      alert(`Reset to base failed: ${err.message || err}`);
    }
  } finally {
    updateInspector();
  }
}

async function linkModelRagProfile(profileId, ragProfileId) {
  try {
    const cleanProfileId = String(profileId || "").trim();
    if (cleanProfileId) {
      delete state.modelRagDraftLinks[cleanProfileId];
    }
    const payload = await callJson("/api/models/link-rag", {
      method: "POST",
      body: JSON.stringify({
        profile_id: cleanProfileId || profileId,
        rag_profile_id: ragProfileId || null,
      }),
    });
    state.lastResponse = payload;
    applyModelStatus(payload, false);
  } catch (err) {
    alert(`Link RAG profile failed: ${err.message || err}`);
  } finally {
    updateInspector();
  }
}

function setRagFileDetails(details) {
  if (el.ragFileDetails) {
    el.ragFileDetails.textContent = pretty(details || {});
  }
}

function renderRagFilesTable() {
  if (!el.ragFilesTableBody) return;
  const files = Array.isArray(state.ragFiles) ? state.ragFiles : [];
  if (files.length === 0) {
    el.ragFilesTableBody.innerHTML = `
      <tr>
        <td colspan="5">No RAG files found. Upload files or index paths.</td>
      </tr>
    `;
    return;
  }

  const rows = files
    .map((file) => {
      const path = String(file.path || "");
      const selectedClass = normalizePath(path) === normalizePath(state.selectedRagPath || "") ? "is-selected" : "";
      const statusLabel = String(file.status || "uploaded");
      const actions = `
        <div class="mini-actions">
          <button class="btn mini subtle icon-open" data-rag-action="details" data-path="${escapeAttr(path)}">Details</button>
          <button class="btn mini subtle icon-open" data-rag-action="use-path" data-path="${escapeAttr(path)}">Use Path</button>
          <button class="btn mini warn icon-delete" data-rag-action="remove" data-path="${escapeAttr(path)}">Remove</button>
          <button class="btn mini accent icon-add" data-rag-action="train" data-path="${escapeAttr(path)}">Add to Training</button>
        </div>
      `;
      return `
        <tr class="${selectedClass}" data-rag-path="${escapeAttr(path)}">
          <td class="file-cell">
            <div>${escapeHtml(file.name || path || "unknown")}</div>
            <div class="small-note">${escapeHtml(path)}</div>
          </td>
          <td>${statusPill(statusLabel, statusTone(statusLabel))}</td>
          <td>${Number(file.chunk_count || 0)}</td>
          <td>${formatBytes(file.size_bytes || 0)}</td>
          <td>${actions}</td>
        </tr>
      `;
    })
    .join("");

  el.ragFilesTableBody.innerHTML = rows;
}

async function refreshRagFiles() {
  try {
    const payload = await callJson("/api/rag/files");
    state.ragFiles = Array.isArray(payload.files) ? payload.files : [];
    renderRagFilesTable();

    const summary = payload.summary || {};
    state.lastResponse = {
      ...(state.lastResponse || {}),
      rag_files_summary: summary,
    };

    if (!state.selectedRagPath && state.ragFiles.length > 0) {
      state.selectedRagPath = state.ragFiles[0].path;
      await loadRagFileDetails(state.selectedRagPath);
      return;
    }

    if (state.selectedRagPath) {
      const match = state.ragFiles.find(
        (file) => normalizePath(file.path) === normalizePath(state.selectedRagPath)
      );
      if (!match) {
        state.selectedRagPath = null;
        setRagFileDetails({
          message: "No file selected.",
          hint: "Click a row or 'Details' to inspect chunk previews.",
        });
      }
    }
  } catch (err) {
    state.ragFiles = [];
    if (el.ragFilesTableBody) {
      el.ragFilesTableBody.innerHTML = `
        <tr>
          <td colspan="5">Failed to load RAG files: ${escapeHtml(String(err.message || err))}</td>
        </tr>
      `;
    }
  } finally {
    updateRagContextControl();
    renderRagProfileSourcesTable();
    updateInspector();
  }
}

async function loadRagFileDetails(path) {
  const value = String(path || "").trim();
  if (!value) return;
  state.selectedRagPath = value;
  renderRagFilesTable();
  setRagFileDetails({
    loading: true,
    path: value,
  });

  try {
    const details = await callJson(
      `/api/rag/file/details?path=${encodeURIComponent(value)}&chunk_limit=6`
    );
    setRagFileDetails(details);
    state.lastResponse = details;
  } catch (err) {
    setRagFileDetails({ error: String(err.message || err), path: value });
  } finally {
    updateInspector();
  }
}

function addRagPath(path) {
  const value = String(path || "").trim();
  if (!value) return;
  el.ragPathsInput.value = mergeUniquePaths(el.ragPathsInput.value, [value]);
}

async function exportRagToTraining(path) {
  const sourcePath = String(path || "").trim();
  if (!sourcePath) return;

  const suggested = datasetStemFromPath(sourcePath);
  const datasetName = (prompt("Training dataset name:", suggested) || "").trim();
  if (!datasetName) return;

  try {
    const payload = await callJson("/api/rag/export-train", {
      method: "POST",
      body: JSON.stringify({
        paths: [sourcePath],
        dataset_name: datasetName,
        chunk_size: Number(el.chunkSizeInput.value || 220),
        chunk_overlap: Number(el.chunkOverlapInput.value || 40),
        append: false,
      }),
    });

    el.trainDatasetInput.value = payload.dataset_path || "";
    state.lastResponse = payload;
    updateInspector();
    setActiveTab("training");
    alert(
      `Prepared training dataset: ${payload.dataset_path}\nRows: ${payload.rows_written || 0}\nSources: ${
        payload.source_count || 0
      }`
    );
  } catch (err) {
    alert(`RAG -> training export failed: ${err.message || err}`);
  }
}

async function removeRagFile(path) {
  const sourcePath = String(path || "").trim();
  if (!sourcePath) return;
  const file = (state.ragFiles || []).find((item) => normalizePath(item.path) === normalizePath(sourcePath));
  const name = file?.name || sourcePath;
  const message = `Remove '${name}' from RAG?\n\nThis will de-index it and delete the file if it is inside the active profile upload directory.`;
  if (!confirm(message)) return;

  try {
    const payload = await callJson("/api/rag/files/remove", {
      method: "POST",
      body: JSON.stringify({
        paths: [sourcePath],
        delete_files: true,
      }),
    });
    state.lastResponse = payload;
    if (payload?.rag_profiles) {
      applyRagProfilesStatus(payload.rag_profiles);
    }
    if (normalizePath(state.selectedRagPath || "") === normalizePath(sourcePath)) {
      state.selectedRagPath = null;
      setRagFileDetails({
        message: "File removed from RAG list.",
        path: sourcePath,
      });
    }
    await refreshRag();
    await refreshRagFiles();

    const removedChunks = Number(payload?.removed_index?.removed_chunks || 0);
    const deletedCount = Array.isArray(payload?.deleted_files) ? payload.deleted_files.length : 0;
    const skipped = Array.isArray(payload?.skipped_files) ? payload.skipped_files : [];
    const skippedMsg =
      skipped.length > 0 ? `\nSkipped: ${skipped.map((row) => `${row.path} (${row.reason})`).join(", ")}` : "";
    alert(`Removed from index: ${removedChunks} chunk(s).\nDeleted files: ${deletedCount}.${skippedMsg}`);
  } catch (err) {
    alert(`RAG remove failed: ${err.message || err}`);
  } finally {
    updateInspector();
  }
}

function selectedRagProfileId() {
  return String(el.ragProfileSelect?.value || "").trim();
}

function resetRagProfileDraft() {
  state.ragProfileDraftMode = true;
  if (el.ragProfileSelect) {
    el.ragProfileSelect.value = "";
  }
  if (el.ragProfileNameInput) {
    el.ragProfileNameInput.value = "";
  }
  if (el.ragProfileDescriptionInput) {
    el.ragProfileDescriptionInput.value = "";
  }
  renderRagProfileSourcesTable();
  updateRagProfileControls();
  if (el.ragProfileNameInput) {
    el.ragProfileNameInput.focus();
  }
}

async function activateSelectedRagProfile() {
  const profileId = selectedRagProfileId();
  if (!profileId) {
    alert("Select a RAG profile first.");
    return;
  }
  try {
    const payload = await callJson("/api/rag/profiles/activate", {
      method: "POST",
      body: JSON.stringify({ profile_id: profileId }),
    });
    state.lastResponse = payload;
    state.ragProfileDraftMode = false;
    applyRagProfilesStatus(payload);
  } catch (err) {
    alert(`Activate RAG profile failed: ${err.message || err}`);
  } finally {
    updateInspector();
  }
}

async function saveRagProfile() {
  const name = String(el.ragProfileNameInput?.value || "").trim();
  const description = String(el.ragProfileDescriptionInput?.value || "").trim();
  if (!name) {
    alert("RAG profile name is required.");
    return;
  }

  const profileId = selectedRagProfileId();
  const existingProfile = findRagProfile(profileId);
  const isUpdate = !!existingProfile && !state.ragProfileDraftMode;
  try {
    let payload;
    if (isUpdate) {
      payload = await callJson("/api/rag/profiles/update", {
        method: "POST",
        body: JSON.stringify({
          profile_id: profileId,
          name,
          description,
        }),
      });
    } else {
      payload = await callJson("/api/rag/profiles/create", {
        method: "POST",
        body: JSON.stringify({
          name,
          description,
          source_paths: [],
          set_active: true,
        }),
      });
    }
    state.lastResponse = payload;
    state.ragProfileDraftMode = false;
    applyRagProfilesStatus(payload);
    await refreshModels(false);
  } catch (err) {
    alert(`${isUpdate ? "Save" : "Create"} RAG profile failed: ${err.message || err}`);
  } finally {
    updateInspector();
  }
}

async function removeSelectedRagProfile() {
  const profileId = selectedRagProfileId();
  const profile = findRagProfile(profileId);
  if (!profileId || !profile) {
    alert("Select a RAG profile first.");
    return;
  }
  if (!confirm(`Remove RAG profile '${profile.name || profile.id}'?`)) return;

  try {
    const payload = await callJson("/api/rag/profiles/remove", {
      method: "POST",
      body: JSON.stringify({ profile_id: profileId }),
    });
    state.lastResponse = payload;
    state.ragProfileDraftMode = false;
    const ragStatus = getRagProfilesPayload(payload) || payload?.rag_profiles || null;
    if (ragStatus) {
      applyRagProfilesStatus(ragStatus);
    } else {
      await refreshRagProfiles();
    }
    if (payload?.models) {
      applyModelStatus(payload.models, false);
    } else {
      await refreshModels(false);
    }
  } catch (err) {
    alert(`Remove RAG profile failed: ${err.message || err}`);
  } finally {
    updateInspector();
  }
}

async function setRagProfileSources(profileId, paths, mode) {
  const cleanProfileId = String(profileId || "").trim();
  const cleanPaths = (paths || []).map((item) => String(item || "").trim()).filter(Boolean);
  if (!cleanProfileId) {
    alert("Select a RAG profile first.");
    return;
  }
  if (cleanPaths.length === 0) {
    alert("No source paths provided.");
    return;
  }

  try {
    const payload = await callJson("/api/rag/profiles/sources", {
      method: "POST",
      body: JSON.stringify({
        profile_id: cleanProfileId,
        paths: cleanPaths,
        mode: mode || "add",
      }),
    });
    state.lastResponse = payload;
    applyRagProfilesStatus(payload);
  } catch (err) {
    alert(`Update profile sources failed: ${err.message || err}`);
  } finally {
    updateInspector();
  }
}

async function addSelectedFileToProfile() {
  const profileId = selectedRagProfileId();
  if (!profileId) {
    alert("Select a RAG profile first.");
    return;
  }

  const selectedPath = String(state.selectedRagPath || "").trim();
  if (!selectedPath) {
    alert("Select a RAG file in the table first.");
    return;
  }
  await setRagProfileSources(profileId, [selectedPath], "add");
}

async function removeSelectedFileFromProfile() {
  const profileId = selectedRagProfileId();
  if (!profileId) {
    alert("Select a RAG profile first.");
    return;
  }
  const selectedPath = String(state.selectedRagPath || "").trim();
  if (!selectedPath) {
    alert("Select a RAG file in the table first.");
    return;
  }
  await setRagProfileSources(profileId, [selectedPath], "remove");
}

function buildChatPayload() {
  const seedRaw = el.seedInput.value.trim();
  const resolvedRagProfileId = effectiveRagProfileIdForChat();
  const ragEnabled = hasAvailableRagContext() && !!el.ragEnabled?.checked;
  const chatMessages = state.messages.filter((m) => {
    if (!(m.role === "user" || m.role === "assistant")) return false;
    return String(m.content || "").trim().length > 0;
  });
  return {
    messages: chatMessages,
    system_prompt: String(el.systemPromptInput.value || ""),
    temperature: Number(el.temperatureInput.value || 0.8),
    top_p: Number(el.topPInput.value || 0.95),
    max_tokens: Number(el.maxTokensInput.value || 512),
    repeat_penalty: Number(el.repeatPenaltyInput.value || 1.05),
    seed: seedRaw ? Number(seedRaw) : null,
    rag_enabled: ragEnabled,
    rag_top_k: Number(el.ragTopKInput.value || 4),
    rag_profile_id: ragEnabled ? resolvedRagProfileId || null : null,
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
    let detail = dataText;
    try {
      const payload = JSON.parse(dataText);
      detail = payload?.error || payload?.detail || detail;
    } catch {
      // keep raw text
    }
    throw new Error(String(detail || "stream error"));
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
    state.lastTokenUsage = parsed.usage;
    state.lastResponse = { ...(state.lastResponse || {}), usage: parsed.usage };
    updateInspector();
  }
}

async function requestServerStart() {
  const selectedModelPath = selectedRuntimeModelPath();
  if (!selectedModelPath) {
    throw new Error("Select a runtime model profile first.");
  }
  const payload = {
    model_path: selectedModelPath,
    host: el.serverHostInput.value.trim() || "127.0.0.1",
    port: Number(el.serverPortInput.value || 8080),
    threads: Number(el.threadsInput.value || 8),
    ctx_size: Number(el.ctxSizeInput.value || 2048),
    n_predict: 4096,
    lora_paths: [],
    extra_args: [],
  };
  return await callJson("/api/server/start", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

async function ensureServerReadyForChat() {
  let status = null;
  try {
    status = await callJson("/api/server/status");
    state.serverStatus = status;
  } catch {
    status = state.serverStatus;
  }

  updateRuntimeModelSelect();
  renderModelContextBar();
  updateInspector();

  if (status?.running) {
    return;
  }

  if (state.serverAction) {
    await refreshHealth();
    if (state.serverStatus?.running) {
      return;
    }
  }

  state.serverAction = "starting";
  syncServerButtons(false);
  try {
    const result = await requestServerStart();
    state.serverStatus = result;
  } finally {
    state.serverAction = null;
    await refreshHealth();
  }

  if (!state.serverStatus?.running) {
    throw new Error("llama-server is not running.");
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
  if (!hasAvailableRagContext() || !el.ragEnabled?.checked) return;
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
          profile_id: effectiveRagProfileIdForChat() || null,
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
  autoResizeComposerInput();

  const payload = buildChatPayload();
  state.lastRequest = payload;
  state.lastResponse = null;
  updateInspector();

  const controller = new AbortController();
  state.streamAbort = controller;
  setChatStreamingState(true);

  try {
    await ensureServerReadyForChat();

    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok || !response.body) {
      let detail = `HTTP ${response.status}`;
      try {
        const raw = await response.text();
        if (raw) {
          try {
            const parsed = JSON.parse(raw);
            detail = parsed?.detail || parsed?.error || raw;
          } catch {
            detail = raw;
          }
        }
      } catch {
        // keep HTTP status fallback
      }
      throw new Error(String(detail || `HTTP ${response.status}`));
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
    let detail = String(err?.message || err || "stream failed");
    if (!detail.trim() || /^502:\s*$/i.test(detail.trim())) {
      detail = "Unable to reach llama-server. The server may still be starting; retry in a moment.";
    }
    state.messages[assistantIndex].content += `\n\n[stream error] ${detail}`;
    renderMessages();
  } finally {
    await applyFrontendRagFallback(assistantIndex, text);
    state.streamAbort = null;
    setChatStreamingState(false);
    persistConversationsToStorage();
    updateInspector();
  }
}

async function startServer() {
  if (state.serverAction) return;
  state.serverAction = "starting";
  syncServerButtons(false);
  try {
    const result = await requestServerStart();
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
    await refreshRagFiles();
    await refreshRagProfiles();

    const skipped = Array.isArray(result?.skipped_files) ? result.skipped_files : [];
    if (Number(result?.added_chunks || 0) <= 0) {
      const details =
        skipped.length > 0 ? ` Skipped: ${skipped.map((item) => `${item.file} (${item.reason})`).join(", ")}` : "";
      alert(`No new chunks were indexed.${details}`);
    }
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
    const indexSkipped = Array.isArray(payload.index_skipped_files) ? payload.index_skipped_files : [];
    el.ragPathsInput.value = mergeUniquePaths(el.ragPathsInput.value, savedFiles);
    el.ragUploadInput.value = "";

    state.lastResponse = payload;
    await refreshRag();
    await refreshRagFiles();
    await refreshRagProfiles();
    const addedChunks = Number(payload.indexed?.added_chunks || 0);
    const uploadSkipSummary =
      skipped.length > 0 ? ` Upload skips: ${skipped.map((item) => `${item.file} (${item.reason})`).join(", ")}.` : "";
    const indexSkipSummary =
      indexSkipped.length > 0
        ? ` Index skips: ${indexSkipped.map((item) => `${item.file} (${item.reason})`).join(", ")}.`
        : "";
    if (addedChunks <= 0) {
      alert(
        `Upload completed but no RAG chunks were indexed. Uploaded ${payload.saved_count || 0} file(s).${uploadSkipSummary}${indexSkipSummary}`
      );
    } else {
      alert(
        `Uploaded ${payload.saved_count || 0} file(s). Added ${addedChunks} chunk(s).${uploadSkipSummary}${indexSkipSummary}`
      );
    }
  } catch (err) {
    alert(`RAG upload failed: ${err.message || err}`);
  } finally {
    el.uploadRagBtn.disabled = false;
    updateInspector();
  }
}

async function resetRag() {
  if (!confirm("Reset RAG for the active profile?")) return;
  try {
    const result = await callJson("/api/rag/reset", { method: "POST" });
    state.lastResponse = result;
    state.selectedRagPath = null;
    setRagFileDetails({ message: "RAG index cleared." });
    await refreshRag();
    await refreshRagFiles();
    await refreshRagProfiles();
  } catch (err) {
    alert(`RAG reset failed: ${err.message || err}`);
  }
}

async function startTraining() {
  if (!isTrainingAllowed()) {
    alert("Training is blocked while Base BitNet is active. Activate a custom model profile first.");
    return;
  }
  try {
    const quantizeValue = String(el.trainQuantizeOutputInput?.value || "false").trim().toLowerCase();
    const payload = {
      base_model: el.trainBaseModelInput.value.trim(),
      model_revision: el.trainModelRevisionInput?.value.trim() || "prequantized",
      tokenizer_model: el.trainTokenizerModelInput?.value.trim() || "",
      dataset_path: el.trainDatasetInput.value.trim(),
      output_dir: el.trainOutputDirInput.value.trim(),
      epochs: Number(el.trainEpochsInput.value || 1),
      batch_size: Number(el.trainBatchInput.value || 1),
      grad_accum_steps: Number(el.trainGradAccumInput.value || 8),
      learning_rate: Number(el.trainLrInput.value || 0.0002),
      max_seq_len: Number(el.trainMaxSeqInput.value || 1024),
      quantize_output: quantizeValue === "true" || quantizeValue === "1" || quantizeValue === "yes",
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
    alert("Select a dataset or document file first.");
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
    if (String(payload?.conversion_mode || "") === "document_to_jsonl") {
      alert(`Prepared dataset from document: ${payload.filename} (${Number(payload?.rows || 0)} rows)`);
    } else {
      alert(`Uploaded dataset: ${payload.filename}`);
    }
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

function setupEvents() {
  if (el.themeToggleInput) {
    el.themeToggleInput.addEventListener("change", () => {
      setThemePreference(el.themeToggleInput.checked ? "dark" : "light");
    });
  }
  if (el.workspaceProfileSelect) {
    el.workspaceProfileSelect.addEventListener("change", async () => {
      const profileId = String(el.workspaceProfileSelect.value || "").trim();
      if (!profileId) {
        state.activeWorkspaceProfileId = "";
        _persistWorkspaceProfilesToStorage();
        renderWorkspaceProfileSelect();
        renderTokenUsageBadge();
        renderModelContextBar();
        updateInspector();
        return;
      }
      state.activeWorkspaceProfileId = profileId;
      _persistWorkspaceProfilesToStorage();
      try {
        await applyWorkspaceProfileById(profileId, {
          silent: true,
          resetChat: false,
          normalizeModel: true,
        });
      } catch (err) {
        alert(`Profile apply failed: ${err.message || err}`);
      }
    });
  }
  if (el.createWorkspaceProfileBtn) {
    el.createWorkspaceProfileBtn.addEventListener("click", createWorkspaceProfile);
  }
  if (el.saveWorkspaceProfileBtn) {
    el.saveWorkspaceProfileBtn.addEventListener("click", saveWorkspaceProfile);
  }
  if (el.deleteWorkspaceProfileBtn) {
    el.deleteWorkspaceProfileBtn.addEventListener("click", deleteWorkspaceProfile);
  }
  if (el.newConversationBtn) {
    el.newConversationBtn.addEventListener("click", createConversation);
  }
  if (el.conversationList) {
    el.conversationList.addEventListener("click", (event) => {
      const rawTarget = event.target;
      const origin = rawTarget instanceof Element ? rawTarget : null;
      if (!origin) return;
      const actionEl = origin.closest("[data-conversation-action]");
      if (!(actionEl instanceof HTMLElement)) return;
      const action = String(actionEl.dataset.conversationAction || "").trim();
      const conversationId = String(actionEl.dataset.conversationId || "").trim();
      if (!action || !conversationId) return;
      if (action === "switch") {
        switchConversation(conversationId);
        return;
      }
      if (action === "rename") {
        renameConversation(conversationId);
        return;
      }
      if (action === "delete") {
        deleteConversation(conversationId);
      }
    });
  }
  if (el.sendBtn) {
    el.sendBtn.addEventListener("click", sendMessage);
  }
  if (el.stopBtn) {
    el.stopBtn.addEventListener("click", () => state.streamAbort && state.streamAbort.abort());
  }
  if (el.userInput) {
    el.userInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    });
    el.userInput.addEventListener("input", () => {
      renderTokenUsageBadge();
      autoResizeComposerInput();
    });
  }
  if (el.startServerBtn) {
    el.startServerBtn.addEventListener("click", startServer);
  }
  if (el.stopServerBtn) {
    el.stopServerBtn.addEventListener("click", stopServer);
  }
  if (el.modelPathInput) {
    const onRuntimeModelChange = async () => {
      await applyRuntimeModelSelection();
    };
    el.modelPathInput.addEventListener("change", onRuntimeModelChange);
    el.modelPathInput.addEventListener("input", onRuntimeModelChange);
  }
  if (el.refreshModelsBtn) {
    el.refreshModelsBtn.addEventListener("click", () => refreshModels(false));
  }
  if (el.useBaseModelBtn) {
    el.useBaseModelBtn.addEventListener("click", () => resetModelToBase({ silent: false }));
  }
  if (el.modelProfileNameInput) {
    el.modelProfileNameInput.addEventListener("input", updateResolvedModelPathPreview);
  }
  if (el.modelProfileDirInput) {
    el.modelProfileDirInput.addEventListener("input", updateResolvedModelPathPreview);
  }
  if (el.browseModelDirBtn) {
    el.browseModelDirBtn.addEventListener("click", pickModelDirectory);
  }
  if (el.addModelProfileBtn) {
    el.addModelProfileBtn.addEventListener("click", registerModelProfile);
  }
  if (el.modelsTableBody) {
    el.modelsTableBody.addEventListener("change", (event) => {
      const rawTarget = event.target;
      const origin = rawTarget instanceof Element ? rawTarget : null;
      if (!origin) return;
      const select = origin.closest(".model-rag-select");
      if (!(select instanceof HTMLSelectElement)) return;

      const profileId =
        String(select.dataset.profileId || "").trim() ||
        String(select.closest("tr[data-profile-id]")?.getAttribute("data-profile-id") || "").trim();
      if (!profileId) return;

      const selectedRagId = String(select.value || "").trim();
      const backendProfile = (state.models?.profiles || []).find((item) => String(item?.id || "") === profileId);
      const backendLinkedRagId = String(backendProfile?.rag_profile_id || "");

      if (selectedRagId === backendLinkedRagId) {
        delete state.modelRagDraftLinks[profileId];
      } else {
        state.modelRagDraftLinks[profileId] = selectedRagId;
      }
      renderModelsTable();
      updateInspector();
    });

    el.modelsTableBody.addEventListener("click", (event) => {
      const rawTarget = event.target;
      const origin = rawTarget instanceof Element ? rawTarget : null;
      if (!origin) return;
      const actionButton = origin.closest("[data-model-action]");
      if (!(actionButton instanceof HTMLElement)) return;

      const action = actionButton.dataset.modelAction;
      const profileId = actionButton.dataset.profileId;
      if (!action || !profileId) return;
      if (action === "activate") {
        activateModelProfile(profileId, { suppressExternalAlert: true });
        return;
      }
      if (action === "link-rag") {
        const row = actionButton.closest("tr[data-profile-id]");
        const select = row ? row.querySelector(".model-rag-select") : null;
        const ragProfileId = select instanceof HTMLSelectElement ? String(select.value || "").trim() : "";
        linkModelRagProfile(profileId, ragProfileId || null);
        return;
      }
      if (action === "remove") {
        removeModelProfile(profileId);
      }
    });
  }
  if (el.indexRagBtn) {
    el.indexRagBtn.addEventListener("click", indexRag);
  }
  if (el.uploadRagBtn) {
    el.uploadRagBtn.addEventListener("click", uploadRagFiles);
  }
  if (el.resetRagBtn) {
    el.resetRagBtn.addEventListener("click", resetRag);
  }
  if (el.ragProfileSelect) {
    el.ragProfileSelect.addEventListener("change", async () => {
      const selectedId = selectedRagProfileId();
      state.ragProfileDraftMode = !findRagProfile(selectedId);
      syncRagProfileEditorFromSelection();
      if (selectedId && selectedId !== activeRagProfileId()) {
        try {
          const payload = await callJson("/api/rag/profiles/activate", {
            method: "POST",
            body: JSON.stringify({ profile_id: selectedId }),
          });
          state.ragProfileDraftMode = false;
          applyRagProfilesStatus(payload);
        } catch (err) {
          alert(`Activate RAG profile failed: ${err.message || err}`);
        } finally {
          updateInspector();
        }
      }
      if (selectedId && selectedId !== String(state.activeWorkspaceProfileId || "").trim()) {
        state.activeWorkspaceProfileId = selectedId;
        renderWorkspaceProfileSelect();
        _persistWorkspaceProfilesToStorage();
        try {
          await applyWorkspaceProfileById(selectedId, {
            silent: true,
            resetChat: false,
            normalizeModel: true,
            activateRag: false,
          });
        } catch (err) {
          alert(`Profile apply failed: ${err.message || err}`);
        }
      }
    });
  }
  if (el.activateRagProfileBtn) {
    el.activateRagProfileBtn.addEventListener("click", activateSelectedRagProfile);
  }
  if (el.newRagProfileBtn) {
    el.newRagProfileBtn.addEventListener("click", resetRagProfileDraft);
  }
  if (el.saveRagProfileBtn) {
    el.saveRagProfileBtn.addEventListener("click", saveRagProfile);
  }
  if (el.removeRagProfileBtn) {
    el.removeRagProfileBtn.addEventListener("click", removeSelectedRagProfile);
  }
  if (el.addSelectedToProfileBtn) {
    el.addSelectedToProfileBtn.addEventListener("click", addSelectedFileToProfile);
  }
  if (el.removeSelectedFromProfileBtn) {
    el.removeSelectedFromProfileBtn.addEventListener("click", removeSelectedFileFromProfile);
  }
  if (el.ragProfileSourcesTableBody) {
    el.ragProfileSourcesTableBody.addEventListener("click", (event) => {
      const rawTarget = event.target;
      const origin = rawTarget instanceof Element ? rawTarget : null;
      if (!origin) return;
      const actionButton = origin.closest("[data-rag-profile-action]");
      if (!(actionButton instanceof HTMLElement)) return;
      const action = actionButton.dataset.ragProfileAction;
      const path = actionButton.dataset.path;
      if (action === "remove-source" && path) {
        const profileId = selectedRagProfileId();
        setRagProfileSources(profileId, [path], "remove");
      }
    });
  }
  if (el.ragProfileNameInput) {
    el.ragProfileNameInput.addEventListener("input", () => {
      if (!hasSelectedRagProfile()) {
        state.ragProfileDraftMode = true;
      }
      updateRagProfileControls();
    });
  }
  if (el.ragProfileDescriptionInput) {
    el.ragProfileDescriptionInput.addEventListener("input", () => {
      if (!hasSelectedRagProfile()) {
        state.ragProfileDraftMode = true;
      }
      updateRagProfileControls();
    });
  }
  if (el.ragFilesTableBody) {
    el.ragFilesTableBody.addEventListener("click", (event) => {
      const rawTarget = event.target;
      const origin = rawTarget instanceof Element ? rawTarget : null;
      if (!origin) return;

      const actionButton = origin.closest("[data-rag-action]");
      if (actionButton instanceof HTMLElement) {
        const action = actionButton.dataset.ragAction;
        const path = actionButton.dataset.path;
        if (action && path) {
          if (action === "details") {
            loadRagFileDetails(path);
            return;
          }
          if (action === "use-path") {
            addRagPath(path);
            return;
          }
          if (action === "remove") {
            removeRagFile(path);
            return;
          }
          if (action === "train") {
            exportRagToTraining(path);
            return;
          }
        }
      }

      const row = origin.closest("tr[data-rag-path]");
      if (!row) return;
      const rowPath = row.getAttribute("data-rag-path");
      if (!rowPath) return;
      loadRagFileDetails(rowPath);
    });
  }
  if (el.uploadTrainBtn) {
    el.uploadTrainBtn.addEventListener("click", uploadTrainingDataset);
  }
  if (el.trainBaseModelInput) {
    el.trainBaseModelInput.addEventListener("input", renderModelContextBar);
  }
  if (el.systemPromptInput) {
    el.systemPromptInput.addEventListener("input", () => {
      renderTokenUsageBadge();
      persistSystemPrompt();
      syncActiveWorkspaceProfileFromUi({ render: false, persist: true });
    });
  }
  if (el.temperatureInput) {
    el.temperatureInput.addEventListener("input", () => {
      syncActiveWorkspaceProfileFromUi({ render: false, persist: true });
    });
  }
  if (el.topPInput) {
    el.topPInput.addEventListener("input", () => {
      syncActiveWorkspaceProfileFromUi({ render: false, persist: true });
    });
  }
  if (el.maxTokensInput) {
    el.maxTokensInput.addEventListener("input", () => {
      renderTokenUsageBadge();
      syncActiveWorkspaceProfileFromUi({ render: false, persist: true });
    });
  }
  if (el.repeatPenaltyInput) {
    el.repeatPenaltyInput.addEventListener("input", () => {
      syncActiveWorkspaceProfileFromUi({ render: false, persist: true });
    });
  }
  if (el.ragEnabled) {
    el.ragEnabled.addEventListener("change", () => {
      renderTokenUsageBadge();
      syncActiveWorkspaceProfileFromUi({ render: false, persist: true });
    });
  }
  if (el.ragTopKInput) {
    el.ragTopKInput.addEventListener("input", () => {
      renderTokenUsageBadge();
      syncActiveWorkspaceProfileFromUi({ render: false, persist: true });
    });
  }
  if (el.seedInput) {
    el.seedInput.addEventListener("input", () => {
      syncActiveWorkspaceProfileFromUi({ render: false, persist: true });
    });
  }
  if (el.createRawDatasetBtn) {
    el.createRawDatasetBtn.addEventListener("click", createTrainingDatasetFromRaw);
  }
  if (el.startTrainBtn) {
    el.startTrainBtn.addEventListener("click", startTraining);
  }
  if (el.stopTrainBtn) {
    el.stopTrainBtn.addEventListener("click", stopTraining);
  }
  if (el.clearChatBtn) {
    el.clearChatBtn.addEventListener("click", () => {
      if (state.streamAbort) return;
      setActiveConversationMessages([]);
      state.lastRetrieval = [];
      state.lastTokenUsage = null;
      updateInspector();
    });
  }
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      el.userInput.value = chip.dataset.prompt || "";
      autoResizeComposerInput();
      el.userInput.focus();
    });
  });
  for (const button of el.tabButtons) {
    button.addEventListener("click", () => setActiveTab(button.dataset.tab));
  }
  for (const button of el.settingsSubtabButtons) {
    button.addEventListener("click", () => setActiveSettingsSubtab(button.dataset.settingsSubtab));
  }
  for (const button of el.ragSubtabButtons) {
    button.addEventListener("click", () => setActiveRagSubtab(button.dataset.ragSubtab));
  }
}

async function boot() {
  applyTheme(getPreferredTheme());
  loadWorkspaceProfilesFromStorage();
  loadConversationsFromStorage();
  loadSystemPromptFromStorage();
  ensureConversationState();
  setupEvents();
  renderWorkspaceProfileSelect();
  renderConversationList();
  renderMessages();
  setActiveSettingsSubtab(state.settingsSubtab);
  setActiveRagSubtab(state.ragSubtab);
  if (el.modelProfileDirInput && !String(el.modelProfileDirInput.value || "").trim()) {
    const runtimeDir = dirnamePath(state.runtimeModelPath || "");
    if (runtimeDir) {
      el.modelProfileDirInput.value = runtimeDir;
    }
  }
  updateResolvedModelPathPreview();
  updateRuntimeModelSelect();
  renderModelContextBar();
  updateTrainingControls();
  renderTokenUsageBadge();
  autoResizeComposerInput();
  setChatStreamingState(false);
  updateRagContextControl();
  setRagFileDetails({
    message: "No file selected.",
    hint: "Click a RAG file row to inspect metadata and chunk previews.",
  });
  syncServerButtons(false);
  await refreshHealth();
  await refreshModels(true);
  await refreshRagProfiles();
  if (state.activeWorkspaceProfileId) {
    try {
      await applyWorkspaceProfileById(state.activeWorkspaceProfileId, {
        silent: true,
        resetChat: false,
        normalizeModel: true,
      });
    } catch (err) {
      console.warn("Profile auto-apply failed:", err);
    }
  }
  if (el.modelProfileDirInput && !String(el.modelProfileDirInput.value || "").trim()) {
    const runtimeDir = dirnamePath(selectedRuntimeModelPath());
    if (runtimeDir) {
      el.modelProfileDirInput.value = runtimeDir;
    }
  }
  updateResolvedModelPathPreview();
  await refreshRag();
  await refreshRagFiles();
  await refreshTraining();
  // pushMessage(
  //   "system",
  //   "Command Deck ready. Start llama-server, optionally index docs for RAG, then send prompts."
  // );
  setInterval(refreshHealth, 5000);
  setInterval(() => refreshModels(false), 12000);
  setInterval(refreshRagProfiles, 12000);
  setInterval(refreshRag, 10000);
  setInterval(refreshRagFiles, 12000);
  setInterval(refreshTraining, 8000);
}

boot();
