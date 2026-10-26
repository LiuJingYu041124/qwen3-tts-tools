let current = null;
let projects = [];
let timer = null;
let projectFilter = "all";
let selectedRoleId = null;
let cueRoleFilter = "all";
let cueSearchQuery = "";
let lastJobs = {};
let toastTimer = null;
const previewJobKeys = new Set();
const referenceTranscriptDrafts = new Map();
const transcriptConfirmKeys = new Set();
const dismissedImportGuides = new Set();
let previewContext = null;

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];

// 批量选择状态
let selectedCues = new Set();

function visibleCueRows() {
  return Array.from(document.querySelectorAll('.cue-strip[data-cue-id]'));
}

function syncCueSelectionStyles() {
  visibleCueRows().forEach(row => {
    row.classList.toggle('batch-selected', selectedCues.has(Number(row.dataset.cueId)));
  });
}

function updateBatchButton() {
  const batchButton = document.getElementById('batchBtn');
  const selectButton = document.getElementById('selectAllBtn');
  const rows = visibleCueRows();
  const selectedVisible = rows.filter(row => selectedCues.has(Number(row.dataset.cueId))).length;
  if (batchButton) batchButton.textContent = selectedCues.size ? `批量调整（${selectedCues.size}）` : '批量调整';
  const deleteButton = document.getElementById('batchDeleteBtn');
  if (deleteButton) {
    deleteButton.disabled = !selectedCues.size;
    deleteButton.textContent = selectedCues.size ? `删除（${selectedCues.size}）` : '删除';
  }
  if (selectButton) selectButton.textContent = rows.length && selectedVisible === rows.length ? '取消全选' : '全选';
  syncCueSelectionStyles();
}

function toggleSelectAll() {
  const rows = visibleCueRows();
  const allSelected = rows.length > 0 && rows.every(row => selectedCues.has(Number(row.dataset.cueId)));
  rows.forEach(row => {
    const cueId = Number(row.dataset.cueId);
    if (allSelected) selectedCues.delete(cueId);
    else selectedCues.add(cueId);
  });
  updateBatchButton();
}

function invertCueSelection() {
  visibleCueRows().forEach(row => {
    const cueId = Number(row.dataset.cueId);
    if (selectedCues.has(cueId)) selectedCues.delete(cueId);
    else selectedCues.add(cueId);
  });
  updateBatchButton();
}

function toggleCueSelection(cueId) {
  cueId = Number(cueId);
  if (selectedCues.has(cueId)) selectedCues.delete(cueId);
  else selectedCues.add(cueId);
  updateBatchButton();
}

function toggleBatchPanel() {
  const panel = document.getElementById('batchPanel');
  if (!panel) return;
  panel.hidden = !panel.hidden;
}

async function deleteSelectedCues() {
  if (!current?.id || !selectedCues.size) return showToast('请先选择要删除的台词', true);
  const count = selectedCues.size;
  if (!confirm(`确定永久删除选中的 ${count} 条台词吗？整轨需要重新生成。`)) return;
  try {
    const result = await api(`/api/projects/${current.id}/cues/batch`, {
      method: 'DELETE', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({cue_ids: Array.from(selectedCues)}),
    });
    current = result.project;
    selectedCues.clear();
    render();
    await refreshProjects();
    showToast(`已删除 ${result.deleted} 条台词`);
  } catch (error) { showToast(error.message, true); }
}

function pronunciationRulesFromText(value) {
  return String(value || '').split(/\n+/).map(line => line.trim()).filter(Boolean).map(line => {
    const parts = line.split(/(?:=>|→|=)/);
    return {source: (parts.shift() || '').trim(), spoken: parts.join('=').trim()};
  }).filter(item => item.source && item.spoken);
}

async function setCuePronunciation(index) {
  const cue = current?.cues?.[index];
  if (!cue) return;
  const selected = window.getSelection()?.toString().trim();
  const source = prompt('请输入需要指定读音的原词或多音字', selected || cue.text || '');
  if (!source) return;
  const existing = (current.settings?.pronunciation_lexicon || []).find(item => item.source === source);
  const spoken = prompt(`“${source}”实际读作（请用同音汉字或自然文字表示）`, existing?.spoken || '');
  if (!spoken) return;
  const rules = [...(current.settings?.pronunciation_lexicon || [])].filter(item => item.source !== source);
  rules.push({source: source.trim(), spoken: spoken.trim()});
  try {
    current = await api(`/api/projects/${current.id}/pronunciation-lexicon`, {
      method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({rules}),
    });
    renderCues(); renderSettings(); renderStatus(); renderTransport();
    showToast(`已设置：${source} → ${spoken}，试听和整轨将使用新读音`);
  } catch (error) { showToast(error.message, true); }
}

function cuePronunciationSummary(cue) {
  const hits = (current?.settings?.pronunciation_lexicon || []).filter(item => String(cue?.text || '').includes(item.source));
  return hits.map(item => `${item.source}→${item.spoken}`).join('；');
}

async function savePronunciationLexicon() {
  try {
    const rules = pronunciationRulesFromText(document.getElementById('pronunciationLexicon')?.value);
    current = await api(`/api/projects/${current.id}/pronunciation-lexicon`, {
      method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({rules}),
    });
    renderSettings(); renderStatus(); renderTransport();
    showToast(`已保存 ${rules.length} 条发音规则，请重新生成`);
  } catch (error) { showToast(error.message, true); }
}

async function deleteSelectedCues() {
  if (!current?.id || !selectedCues.size) return showToast('请先选择要删除的台词', true);
  const count = selectedCues.size;
  if (!confirm(`确定永久删除选中的 ${count} 条台词吗？整轨需要重新生成。`)) return;
  try {
    const result = await api(`/api/projects/${current.id}/cues/batch`, {
      method: 'DELETE', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({cue_ids: Array.from(selectedCues)}),
    });
    current = result.project;
    selectedCues.clear();
    render();
    await refreshProjects();
    showToast(`已删除 ${result.deleted} 条台词`);
  } catch (error) { showToast(error.message, true); }
}

function pronunciationRulesFromText(value) {
  return String(value || '').split(/\n+/).map(line => line.trim()).filter(Boolean).map(line => {
    const parts = line.split(/(?:=>|→|=)/);
    return {source: (parts.shift() || '').trim(), spoken: parts.join('=').trim()};
  }).filter(item => item.source && item.spoken);
}

async function applyBatchSpeedGap() {
  const status = document.getElementById('batchStatus');
  if (!current?.id) {
    if (status) status.textContent = '请先打开项目';
    return;
  }
  if (!selectedCues.size) {
    if (status) status.textContent = '请先选择至少一条台词';
    return;
  }
  const speed = Number(document.getElementById('batchSpeed')?.value);
  const gap = Number(document.getElementById('batchGap')?.value);
  if (!Number.isFinite(speed) || !Number.isFinite(gap)) {
    if (status) status.textContent = '请输入有效的语速和间隔';
    return;
  }
  try {
    const response = await fetch(`/api/projects/${current.id}/cues/batch-speed-gap`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({cue_ids: Array.from(selectedCues), speed, gap_after: gap, selected_all: false})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '批量更新失败');
    (current.cues || []).forEach(cue => {
      if (selectedCues.has(Number(cue.id))) {
        cue.speed = speed;
        cue.gap_after = gap;
      }
    });
    if (status) status.textContent = `已更新 ${result.updated} 条台词`;
    document.getElementById('batchPanel').hidden = true;
    renderCues();
  } catch (error) {
    if (status) status.textContent = error.message;
  }
}

function setHidden(target, hidden) {
  const node = typeof target === "string" ? $(target) : target;
  if (node) node.hidden = Boolean(hidden);
}
function inputValue(target) {
  const node = typeof target === "string" ? $(target) : target;
  return String(node?.value || "").trim();
}
function setInputValue(target, value) {
  const node = typeof target === "string" ? $(target) : target;
  if (node) node.value = value;
}
const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
const icon = name => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
const localVoiceOptions = {
  male_adult: "沉稳成年男声",
  male_young: "明亮年轻男声",
  female_warm: "温暖年轻女声",
  female_brave: "有力量的英气女声",
  girl_soft: "柔软清甜女孩声",
  girl_lively: "明亮活泼女孩声",
};
const localVoiceDescriptions = {
  male_adult: "三十五岁左右的中国成年男性声音，中低音，沉稳可靠，略带粗犷，普通话标准，情绪自然鲜明，不要播音腔。",
  male_young: "二十岁左右的中国年轻男性声音，音色明亮自然，有少年感和好奇心，普通话标准，不要儿童声。",
  female_warm: "二十五岁左右的中国年轻女性声音，温暖爽朗，音色清亮自然，有活力，普通话标准。",
  female_brave: "二十五岁左右的中国年轻女性冒险者声音，有力量、略带英气，情绪鲜明，普通话标准。",
  girl_soft: "一个可爱的中国小女孩声音，年龄感约六到八岁，音色软萌清甜，普通话标准，不要成人腔。",
  girl_lively: "一个活泼勇敢的中国小女孩声音，年龄感约八到十岁，音色明亮俏皮，普通话标准。",
};

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const error = Error(payload?.error || payload?.message || payload || "请求失败");
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function showToast(message, error = false) {
  const toast = $("#toast");
  if (!toast) return;
  toast.textContent = message;
  toast.className = `toast${error ? " error" : ""}`;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 3400);
}

function formatClock(value) {
  const seconds = Math.max(0, Number(value) || 0);
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return `${String(minutes).padStart(2, "0")}:${rest.toFixed(1).padStart(4, "0")}`;
}

function projectMessage(project) {
  const progress = project.tts_progress;
  if (progress?.status === "running") {
    if (progress.current_index && progress.total) return `${progress.stage_label || "处理中"} · ${progress.current_index}/${progress.total}`;
    return progress.stage_label || progress.message || "处理中";
  }
  if (project.pipeline?.status === "done" || project.tts_progress?.status === "done") return "已完成，可试听导出";
  if (project.pipeline?.status === "failed" || project.tts_progress?.status === "failed") return "处理失败，可重试";
  if (project.pipeline?.status === "queued") return project.pipeline.message || "排队中：等待语音任务资源";
  return project.pipeline?.message || project.last_message || project.created_at;
}

async function refreshProjects() {
  projects = await api("/api/projects");
  const filtered = projectFilter === "all" ? projects : projects.filter(project => (project.project_type || "video") === projectFilter);
  const count = $("#projectCount");
  const list = $("#projects");
  if (count) count.textContent = projects.length;
  if (!list) return;
  list.innerHTML = filtered.length ? filtered.map(project => {
    const type = project.project_type || "video";
    return `<div class="project-item ${type} ${current?.id === project.id ? "active" : ""}">
      <button class="project-open" type="button" onclick="loadProject('${project.id}')" aria-label="打开项目 ${esc(project.name)}">
        <i class="project-notch"></i><span><b>${esc(project.name)}</b><small>${esc(projectMessage(project))}</small></span>
      </button>
      <button class="project-delete" type="button" onclick="openDeleteProjectDialog('${project.id}')" aria-label="删除项目 ${esc(project.name)}" title="删除项目">${icon("trash")}</button>
    </div>`;
  }).join("") : `<p class="project-empty">这里还没有${projectFilter === "all" ? "项目" : projectFilter === "voice" ? "脚本配音" : "视频配音"}。</p>`;
}

function setProjectFilter(value) {
  projectFilter = value;
  $$(".filter-tabs button").forEach(button => button.classList.toggle("active", button.dataset.filter === value));
  refreshProjects();
}

function openDeleteProjectDialog(id) {
  const project = projects.find(item => item.id === id);
  const dialog = $("#deleteProjectDialog");
  if (!project || !dialog) return;
  dialog.dataset.projectId = id;
  $("#deleteProjectName").textContent = project.name || "未命名项目";
  $("#deleteProjectState").textContent = "将永久删除项目、源文件副本、参考音频、候选试听、已采用 Take、缓存和最终输出。此操作无法撤销。";
  $("#confirmDeleteProjectButton").disabled = false;
  dialog.showModal();
}

function closeDeleteProjectDialog() {
  const dialog = $("#deleteProjectDialog");
  if (!dialog) return;
  dialog.dataset.projectId = "";
  const button = $("#confirmDeleteProjectButton");
  if (button) button.innerHTML = `${icon("trash")}永久删除`;
  dialog.close();
}

async function confirmDeleteProject() {
  const dialog = $("#deleteProjectDialog");
  const projectId = dialog?.dataset.projectId;
  const button = $("#confirmDeleteProjectButton");
  if (!projectId || !button) return;
  button.disabled = true;
  button.innerHTML = `${icon("wave")}删除中…`;
  try {
    const result = await api(`/api/projects/${projectId}`, {method: "DELETE"});
    if (current?.id === projectId) {
      clearTimeout(timer);
      current = null;
      selectedRoleId = null;
      cueRoleFilter = "all";
      cueSearchQuery = "";
      lastJobs = {};
      previewContext = null;
      previewJobKeys.clear();
      const audio = $("#outputAudio");
      if (audio) {
        audio.pause();
        audio.removeAttribute("src");
        audio.load();
      }
      setHidden("#workspace", true);
      setHidden("#empty", false);
      $("#crumbName").textContent = "尚未打开项目";
      $("#log").textContent = "尚无日志";
      const url = new URL(location.href);
      url.searchParams.delete("project");
      history.replaceState(null, "", url);
    }
    closeDeleteProjectDialog();
    await refreshProjects();
    showToast(`已删除“${result.name}”及其全部生成文件`);
  } catch (error) {
    button.disabled = false;
    button.innerHTML = `${icon("trash")}永久删除`;
    $("#deleteProjectState").textContent = error.message;
    showToast(error.message, true);
  }
}

async function loadProject(id) {
  const previousProjectId = current?.id;
  current = await api(`/api/projects/${id}`);
  previewContext = null;
  if (previousProjectId !== id) {
    cueRoleFilter = "all";
    cueSearchQuery = "";
  }
  selectedRoleId = current.roles?.some(role => role.id === selectedRoleId) ? selectedRoleId : current.roles?.[0]?.id || null;
  const url = new URL(location.href);
  url.searchParams.set("project", id);
  history.replaceState(null, "", url);
  render();
  await refreshProjects();
  poll();
}

function openCreateDialog() {
  updateCreateDocumentOptions();
  $("#createDialog").showModal();
  setTimeout(() => $("#newProjectName").select(), 30);
}

function showCreateDocument(file) {
  $("#createDocumentName").textContent = file?.name || "导入 TXT / Word / PDF";
  if (file && ($("#newProjectName").value === "未命名配音" || !$("#newProjectName").value.trim())) {
    $("#newProjectName").value = file.name.replace(/\.[^.]+$/, "");
  }
  updateCreateDocumentOptions();
}

function importVoiceModeHint(mode) {
  return mode === "clone"
    ? "跳过音色设计与上下文精修。新增角色直接进入授权克隆；上传参考音频并确认逐字稿后，再手动生成。"
    : "导入后自动分析角色音色与逐句情绪，完成后可试听生成。";
}

function updateCreateDocumentOptions() {
  const hasDocument = Boolean($("#createDocumentFile").files[0]);
  setHidden("#createVoiceModeField", !hasDocument);
  const note = $("#createDocumentState");
  note.classList.remove("error");
  note.textContent = hasDocument
    ? importVoiceModeHint(inputValue("#createVoiceMode"))
    : "有文稿就拖进来拆成台词；没有文稿也可直接创建空白对白台。";
}

function updateDocumentImportOptions() {
  const intoCurrent = $("#documentDialog").dataset.target === "current";
  const mode = inputValue("#documentVoiceMode");
  $("#documentDialogHint").textContent = mode === "clone"
    ? "只导入角色与台词，无需等待音色设计"
    : "保留原文顺序，识别角色并拆分台词";
  const note = $("#documentUploadState");
  note.classList.remove("error");
  const importHint = intoCurrent
    ? (isPlaceholderScript(current)
      ? "导入后会替换空白占位台词。"
      : "保留现有台词并追加文稿；同名角色复用原声线与参考音频。")
    : "明确的“角色：台词”会新建角色。";
  note.textContent = `${importHint}${importVoiceModeHint(mode)}`;
}

function documentImportProgress(mode) {
  return mode === "clone"
    ? "正在读取文档、识别角色并拆分台词，不启动音色设计或自动生成……"
    : "正在读取文档，并按正文与配音方向拆分台词、分配情绪和音色，请保持窗口开启……";
}

function documentImportSuccess(imported) {
  const nextStep = imported.voice_mode === "clone"
    ? "已跳过音色设计，可直接配置授权克隆"
    : "正在后台精修音色与情绪";
  return `已导入 ${imported.cue_count || current.cues?.length || 0} 条台词、${imported.role_count || current.roles?.length || 0} 个角色；${nextStep}`;
}

function onCreateDocumentDrag(event, over) {
  event.preventDefault();
  $("#createDocumentDrop").classList.toggle("over", over);
}

function onCreateDocumentDrop(event) {
  event.preventDefault();
  $("#createDocumentDrop").classList.remove("over");
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  $("#createDocumentFile").files = transfer.files;
  showCreateDocument(file);
}

function resetCreateDialog() {
  $("#createDocumentFile").value = "";
  $("#createDocumentName").textContent = "导入 TXT / Word / PDF";
  setInputValue("#createVoiceMode", "design");
  updateCreateDocumentOptions();
  $("#newProjectName").value = "未命名配音";
  setInputValue("#newProjectDirection", "");
  $("#createProjectButton").disabled = false;
}

function openVideoDialog() {
  $("#videoDialog").showModal();
}

function openDocumentDialog(intoCurrent = false) {
  const dialog = $("#documentDialog");
  if (!dialog) {
    showToast("导入窗口加载失败，请刷新页面后重试", true);
    return;
  }
  intoCurrent = Boolean(intoCurrent && current && current.project_type === "voice");
  dialog.dataset.target = intoCurrent ? "current" : "new";
  $("#documentDialogTitle").textContent = intoCurrent ? "导入到当前项目" : "导入文档脚本";
  setHidden("#documentNameField", intoCurrent);
  setHidden("#documentLangField", intoCurrent);
  setInputValue("#documentDirection", intoCurrent ? (current.direction_brief || "") : "");
  updateDocumentImportOptions();
  $("#documentUploadButton").innerHTML = `${icon("upload")}${intoCurrent ? "导入到当前剪辑台" : "导入到对白剪辑台"}`;
  if (typeof dialog.showModal === "function") dialog.showModal();
  else dialog.setAttribute("open", "");
}

function isPlaceholderScript(project) {
  const cues = project?.cues || [];
  if (!cues.length) return true;
  if (cues.length !== 1) return false;
  const text = String(cues[0].text || "").trim();
  return !text || text === "在这里输入第一句台词。";
}

function showSelectedDocument(file) {
  $("#documentFileName").textContent = file?.name || "选择或拖入文稿";
  if (file && !$("#documentProjectName").value) $("#documentProjectName").value = file.name.replace(/\.[^.]+$/, "");
}

function onDocumentDrag(event, over) {
  event.preventDefault();
  $("#documentDrop").classList.toggle("over", over);
}

function onDocumentDrop(event) {
  event.preventDefault();
  $("#documentDrop").classList.remove("over");
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  const input = $("#documentFile");
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  showSelectedDocument(file);
}

async function uploadDocument(event) {
  event.preventDefault();
  const file = $("#documentFile").files[0];
  if (!file) return showToast("请先选择文档", true);
  const intoCurrent = $("#documentDialog").dataset.target === "current" && current?.id;
  const voiceMode = inputValue("#documentVoiceMode") || "design";
  const formData = new FormData();
  formData.append("document", file);
  formData.append("voice_mode", voiceMode);
  formData.append("direction_brief", inputValue("#documentDirection"));
  if (!intoCurrent) {
    formData.append("name", $("#documentProjectName").value.trim());
    formData.append("target_lang", $("#documentLang").value);
  }
  const button = $("#documentUploadButton");
  const note = $("#documentUploadState");
  button.disabled = true;
  note.classList.remove("error");
  note.textContent = documentImportProgress(voiceMode);
  try {
    const url = intoCurrent ? `/api/projects/${current.id}/script-upload` : "/api/script-upload";
    const project = await api(url, {method: "POST", body: formData});
    $("#documentDialog").close();
    button.disabled = false;
    $("#documentFile").value = "";
    $("#documentFileName").textContent = "选择或拖入文稿";
    $("#documentProjectName").value = "";
    setInputValue("#documentDirection", "");
    await loadProject(project.id);
    const imported = current.script_import || project.script_import || {};
    focusImportedVoiceSetup();
    showToast(documentImportSuccess(imported));
    poll();
  } catch (error) {
    button.disabled = false;
    note.classList.add("error");
    note.textContent = error.message;
    showToast(error.message, true);
  }
}

async function createVoiceProject(event) {
  event.preventDefault();
  const file = $("#createDocumentFile").files[0];
  const name = $("#newProjectName").value.trim() || (file ? file.name.replace(/\.[^.]+$/, "") : "未命名配音");
  const language = $("#newProjectLang").value;
  const directionBrief = inputValue("#newProjectDirection");
  const voiceMode = inputValue("#createVoiceMode") || "design";
  const button = $("#createProjectButton");
  const note = $("#createDocumentState");
  button.disabled = true;
  note.classList.remove("error");
  try {
    let project;
    if (file) {
      note.textContent = documentImportProgress(voiceMode);
      const formData = new FormData();
      formData.append("document", file);
      formData.append("name", name);
      formData.append("target_lang", language);
      formData.append("direction_brief", directionBrief);
      formData.append("voice_mode", voiceMode);
      project = await api("/api/script-upload", {method: "POST", body: formData});
    } else {
      project = await api("/api/projects", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({project_type: "voice", name, target_lang: language, direction_brief: directionBrief}),
      });
    }
    $("#createDialog").close();
    resetCreateDialog();
    await loadProject(project.id);
    const imported = current.script_import || project.script_import || {};
    if (file) focusImportedVoiceSetup();
    showToast(file ? documentImportSuccess(imported) : "脚本配音项目已创建");
    if (file) poll();
  } catch (error) {
    button.disabled = false;
    note.classList.add("error");
    note.textContent = error.message;
    showToast(error.message, true);
  }
}

function showSelectedVideo(file) {
  $("#videoFileName").textContent = file?.name || "选择视频文件";
  if (file && !$("#videoProjectName").value) $("#videoProjectName").value = file.name.replace(/\.[^.]+$/, "");
}

async function uploadVideo(event) {
  event.preventDefault();
  const file = $("#videoFile").files[0];
  if (!file) return showToast("请先选择视频", true);
  const formData = new FormData();
  formData.append("video", file);
  formData.append("name", $("#videoProjectName").value.trim());
  formData.append("target_lang", $("#videoLang").value);
  formData.append("tts_engine", "local");
  const button = $("#videoUploadButton");
  button.disabled = true;
  $("#videoUploadState").textContent = "正在上传视频，请保持窗口开启……";
  try {
    const project = await api("/api/auto", {method: "POST", body: formData});
    $("#videoDialog").close();
    button.disabled = false;
    $("#videoUploadState").textContent = "本地 Whisper 识别，Ollama 分析角色，Qwen3-TTS 生成配音。";
    await loadProject(project.id);
  } catch (error) {
    button.disabled = false;
    $("#videoUploadState").textContent = error.message;
    showToast(error.message, true);
  }
}

function render() {
  if (!current) return;
  setHidden("#empty", true);
  setHidden("#workspace", false);
  const type = current.project_type || "video";
  const crumb = $("#crumbName");
  const name = $("#projectName");
  const typeLabel = $("#projectType");
  const source = $("#projectSource");
  if (crumb) crumb.textContent = current.name;
  if (name) name.value = current.name;
  const direction = $("#projectDirection");
  if (direction && document.activeElement !== direction) direction.value = current.direction_brief || "";
  setHidden("#projectDirectionField", type !== "voice");
  if (typeLabel) {
    typeLabel.textContent = type === "voice" ? "脚本配音" : "视频配音";
    typeLabel.className = `type-label${type === "video" ? " video" : ""}`;
  }
  if (source) {
    source.textContent = type === "video"
      ? (current.original_source || current.source)
      : (current.original_source && current.original_source !== "空白脚本"
        ? `${current.original_source} · ${languageName(current.settings?.target_lang)}`
        : `${languageName(current.settings?.target_lang)} · 本地双声线引擎`);
  }
  setHidden("#benchImportButton", type !== "voice");
  setHidden("#emptyImportButton", type !== "voice");
  renderStatus();
  renderTransport();
  renderCues();
  renderRoles();
  renderImportedVoiceGuide();
  renderSettings();
}

function languageName(value) {
  return {zh: "中文", en: "英语", ja: "日语", ko: "韩语"}[value] || value || "中文";
}

function pendingCloneRoles() {
  const used = new Set((current?.cues || []).filter(cue => cue.enabled !== false).map(cue => cue.role));
  return (current?.roles || []).filter(role => role.mode === "clone" && used.has(role.id) && !cloneTranscriptUsable(role));
}

function renderStatus() {
  const pipeline = current.pipeline || {};
  const detail = current.tts_progress || {};
  const detailActive = detail.status === "running" && (pipeline.stage === "tts" || current.project_type === "voice");
  const entries = Object.entries(lastJobs || {}).filter(([key]) => !key.startsWith("preview-"));
  const jobs = entries.map(([, job]) => job);
  const runningJob = jobs.find(job => job.status === "running");
  const queuedJob = jobs.find(job => job.status === "queued");
  const failedJob = entries.filter(([key]) => key !== "direction").map(([, job]) => job).find(job => job.status === "failed");
  let status = failedJob ? "failed" : runningJob ? "running" : queuedJob ? "queued" : (detail.status === "failed" ? "failed" : detail.status === "done" ? "done" : (pipeline.status || "pending"));
  if (current.project_type === "voice" && current.output_audio && !failedJob && !runningJob && !queuedJob) status = "done";
  const hasMeasuredProgress = detailActive && Number.isFinite(Number(detail.percent));
  const percent = status === "done" ? 100 : status === "queued" ? 0 : (hasMeasuredProgress ? Number(detail.percent) : Number(pipeline.percent || 0));
  const queuePosition = Number(queuedJob?.queue_position || 0);
  const queueMessage = queuePosition > 1
    ? `排队中：前面还有 ${queuePosition - 1} 个任务`
    : "排队中：等待语音资源，随后自动开始";
  const progressMessage = detailActive ? (detail.message || detail.stage_label || "") : "";
  const cueProgress = detailActive && detail.current_index && detail.total
    ? ` · 第 ${detail.current_index}/${detail.total} 句`
    : detailActive && detail.total ? ` · 已完成 ${detail.generated || detail.processed || 0}/${detail.total} 句` : "";
  const message = failedJob?.message || (queuedJob ? queueMessage : null) || (runningJob && progressMessage ? `${progressMessage}${cueProgress}` : null) || runningJob?.message || pipeline.message || current.last_message || "等待编辑";
  const strip = $("#statusStrip");
  if (strip) strip.className = `status-strip ${status}`;
  const statusText = $("#statusText");
  if (statusText) statusText.textContent = message;
  const progressBar = $(".status-progress");
  if (progressBar) progressBar.hidden = status === "pending";
  const progress = $("#statusProgress");
  if (progress) {
    progress.style.width = `${percent}%`;
    progress.classList.toggle("indeterminate", detailActive && detail.progress_mode === "indeterminate" && status === "running");
    const progressBar = progress.parentElement;
    if (progressBar) {
      progressBar.setAttribute("aria-valuenow", String(Math.round(percent)));
      progressBar.setAttribute("aria-valuetext", detail.total && detail.current_index
        ? `${Math.round(percent)}%，第 ${detail.current_index}/${detail.total} 句`
        : `${Math.round(percent)}%`);
    }
  }
  const percentLabel = $("#statusPercent");
  if (percentLabel) {
    const remaining = detailActive ? Number(detail.estimated_remaining_seconds || 0) : 0;
    const timeText = remaining > 0 ? ` · 预计还需 ${formatClock(remaining)}` : "";
    percentLabel.textContent = `${percent}%${timeText}`;
    percentLabel.title = detailActive && detail.elapsed_seconds ? `已用时 ${formatClock(detail.elapsed_seconds)}` : "";
  }
  const detailLabel = $("#statusDetail");
  if (detailLabel) {
    const stats = [];
    if (detailActive && detail.total) stats.push(`生成 ${detail.generated || 0}/${detail.total}`);
    if (detailActive && detail.reused) stats.push(`复用 ${detail.reused}`);
    if (detailActive && detail.processed) stats.push(`校验 ${detail.processed}`);
    if (detailActive && detail.needs_review) stats.push(`待复听 ${detail.needs_review}`);
    if (detailActive && detail.failed) stats.push(`失败 ${detail.failed}`);
    detailLabel.textContent = stats.join(" · ");
    detailLabel.hidden = !stats.length;
  }
  setHidden("#retryButton", status !== "failed" || current.project_type === "voice");
  const directing = lastJobs?.direction?.status === "running";
  const generating = entries.some(([key, job]) => key !== "direction" && job.status === "running") || (pipeline.status === "running" && !directing);
  const queued = Boolean(queuedJob);
  const generateButton = $("#generateButton");
  if (generateButton) {
    const pendingClones = pendingCloneRoles();
    generateButton.disabled = Boolean(generating || queued || pendingClones.length);
    generateButton.title = pendingClones.length
      ? `请先确认角色“${pendingClones[0].name || "未命名"}”的参考逐字稿`
      : queued ? queueMessage : "";
    const actionName = current.project_type === "video" ? "生成成片" : "生成整轨";
    generateButton.innerHTML = generating ? `${icon("wave")}生成中…` : queued ? `${icon("wave")}排队中…` : `${icon("wave")}${actionName}`;
  }
  const hasOutput = current.project_type === "voice" ? current.output_audio : current.output;
  setHidden("#downloadButton", !hasOutput);
}

function renderTransport() {
  const audio = $("#outputAudio");
  const hasAudio = current.project_type === "voice" ? Boolean(current.output_audio) : Boolean(current.output);
  const newSource = hasAudio ? `/api/projects/${current.id}/preview?v=${encodeURIComponent(current.updated_at || "")}` : "";
  const loadedSource = audio.dataset.source || "";
  if (newSource && loadedSource !== newSource) {
    const wasPlaying = !audio.paused;
    audio.src = newSource;
    audio.dataset.source = newSource;
    audio.load();
    if (wasPlaying) audio.play().catch(() => {});
  }
  if (!hasAudio) {
    audio.pause();
    audio.removeAttribute("src");
    delete audio.dataset.source;
    $("#transportCurrent").textContent = "00:00.0";
    $("#transportDuration").textContent = "00:00.0";
    $("#transportFill").style.width = "0%";
  }
  setHidden("#videoPreviewButton", current.project_type !== "video" || !current.output);
}

function activeRole() {
  return current.roles?.find(role => role.id === selectedRoleId) || current.roles?.[0] || null;
}

function roleModeLabel(role) {
  return role?.mode === "clone" ? "授权克隆" : "场景造声";
}

function roleModeIcon(role) {
  return role?.mode === "clone" ? "clone" : "spark";
}

function importedRoleIds() {
  const imported = current?.script_import;
  if (!imported) return [];
  const listed = (imported.roles || []).map(role => role.id).filter(Boolean);
  if (listed.length) return listed;
  const directionIds = imported.direction?.cue_ids;
  const wanted = directionIds == null ? null : new Set(directionIds.map(Number));
  const cues = wanted == null
    ? (current?.cues || [])
    : (current?.cues || []).filter(cue => wanted.has(Number(cue.id)));
  return [...new Set(cues.map(cue => cue.role).filter(Boolean))];
}

function importedRoleEntries() {
  const summary = new Map((current?.script_import?.roles || []).map(role => [role.id, role]));
  return importedRoleIds().map(roleId => {
    const role = (current?.roles || []).find(item => item.id === roleId);
    if (!role) return null;
    const cueCount = summary.get(roleId)?.cue_count ?? (current?.cues || []).filter(cue => cue.role === roleId).length;
    return {role, cueCount};
  }).filter(Boolean);
}

function importedRoleState(role) {
  if (role.mode !== "clone") return {key: "design", label: "当前使用场景造声", action: "添加参考配音"};
  if (!role.clone_consent || !role.ref_audio) return {key: "missing", label: "等待上传授权参考配音", action: "继续配置"};
  if (!cloneTranscriptUsable(role)) return {key: "pending", label: "参考音频已上传，逐字稿待确认", action: "核听并确认"};
  return {key: "ready", label: "克隆已就绪", action: "查看克隆"};
}

function renderImportedVoiceGuide() {
  const guide = $("#importedVoiceGuide");
  if (!guide) return;
  const entries = importedRoleEntries();
  const visible = current?.project_type === "voice" && entries.length && !dismissedImportGuides.has(current.id);
  setHidden(guide, !visible);
  if (!visible) return;
  const cloneReady = entries.filter(({role}) => importedRoleState(role).key === "ready").length;
  const cloneInProgress = entries.filter(({role}) => ["missing", "pending"].includes(importedRoleState(role).key)).length;
  $("#importedVoiceGuideSummary").textContent = `${entries.length} 个文稿角色 · ${cloneReady} 个克隆就绪${cloneInProgress ? ` · ${cloneInProgress} 个待完成` : ""}`;
  $("#importedVoiceGuideRoles").innerHTML = entries.map(({role, cueCount}) => {
    const state = importedRoleState(role);
    return `<button class="imported-role-row ${state.key} ${role.id === selectedRoleId ? "active" : ""}" type="button" onclick="openImportedRole('${role.id}')">
      <i></i><span><b>${esc(role.name)}</b><small>${cueCount} 条台词 · ${esc(state.label)}</small></span><em>${esc(state.action)}</em>
    </button>`;
  }).join("");
  const nextButton = $("#configureNextImportedRole");
  const hasActionable = entries.some(({role}) => importedRoleState(role).key !== "ready");
  if (nextButton) {
    nextButton.disabled = !hasActionable;
    nextButton.innerHTML = hasActionable ? `${icon("clone")}配置下一个角色` : `${icon("check")}克隆角色已配置`;
  }
}

function openImportedRole(roleId) {
  const role = (current?.roles || []).find(item => item.id === roleId);
  if (!role) return;
  selectedRoleId = roleId;
  if (role.mode !== "clone") role.mode = "clone";
  renderRoles();
  renderCues();
  renderImportedVoiceGuide();
  setTimeout(() => $("#roleInspector")?.scrollIntoView({behavior: "smooth", block: "start"}), 20);
}

function selectNextImportedRole() {
  const entries = importedRoleEntries();
  const next = entries.find(({role}) => ["missing", "pending"].includes(importedRoleState(role).key))
    || entries.find(({role}) => importedRoleState(role).key === "design")
    || entries[0];
  if (next) openImportedRole(next.role.id);
}

function focusImportedVoiceSetup() {
  dismissedImportGuides.delete(current?.id);
  const entries = importedRoleEntries();
  const next = entries.find(({role}) => importedRoleState(role).key !== "ready") || entries[0];
  if (next) selectedRoleId = next.role.id;
  renderRoles();
  renderImportedVoiceGuide();
  setTimeout(() => $("#importedVoiceGuide")?.scrollIntoView({behavior: "smooth", block: "nearest"}), 30);
}

function dismissImportedVoiceGuide() {
  if (!current?.id) return;
  dismissedImportGuides.add(current.id);
  renderImportedVoiceGuide();
}

function roleOptions(selected) {
  return (current.roles || []).map(role => `<option value="${esc(role.id)}" ${role.id === selected ? "selected" : ""}>${esc(role.name)}</option>`).join("");
}

function cueCountForRole(roleId) {
  return (current?.cues || []).filter(cue => cue.role === roleId && cue.enabled !== false).length;
}

function renderCueFilters() {
  const roles = current.roles || [];
  if (cueRoleFilter !== "all" && !roles.some(role => role.id === cueRoleFilter)) cueRoleFilter = "all";
  const filters = $("#cueRoleFilters");
  if (filters) {
    const allCount = (current.cues || []).filter(cue => cue.enabled !== false).length;
    filters.innerHTML = [
      `<button class="${cueRoleFilter === "all" ? "active" : ""}" type="button" onclick="setCueRoleFilter('all')"><span>全部角色</span><b>${allCount}</b></button>`,
      ...roles.map(role => `<button class="${cueRoleFilter === role.id ? "active" : ""} ${role.mode === "clone" ? "clone" : "design"}" type="button" onclick="setCueRoleFilter('${esc(role.id)}')"><i></i><span>${esc(role.name)}</span><b>${cueCountForRole(role.id)}</b></button>`),
    ].join("");
  }
  const search = $("#cueSearch");
  if (search && document.activeElement !== search) search.value = cueSearchQuery;
  setHidden("#clearCueFilters", cueRoleFilter === "all" && !cueSearchQuery);
}

function visibleCueEntries() {
  const query = cueSearchQuery.trim().toLocaleLowerCase();
  const roleNames = new Map((current.roles || []).map(role => [role.id, role.name]));
  return (current.cues || []).map((cue, index) => ({cue, index})).filter(({cue}) => {
    if (cueRoleFilter !== "all" && cue.role !== cueRoleFilter) return false;
    if (!query) return true;
    return [cue.text, cue.source_text, cue.emotion, roleNames.get(cue.role)].some(value => String(value || "").toLocaleLowerCase().includes(query));
  });
}

function setCueRoleFilter(roleId) {
  cueRoleFilter = roleId;
  renderCues();
  $("#cues")?.scrollTo({top: 0, behavior: "smooth"});
}

function setCueSearch(value) {
  cueSearchQuery = String(value || "");
  renderCues();
}

function clearCueFilters() {
  cueRoleFilter = "all";
  cueSearchQuery = "";
  renderCues();
  $("#cueSearch")?.focus();
}

function showRoleCues(roleId) {
  selectedRoleId = roleId;
  cueRoleFilter = roleId;
  cueSearchQuery = "";
  renderRoles();
  renderImportedVoiceGuide();
  renderCues();
  setTimeout(() => $("#cues")?.scrollTo({top: 0, behavior: "smooth"}), 20);
}

function renderCues() {
  const cues = current.cues || [];
  const entries = visibleCueEntries();
  const enabledCount = cues.filter(cue => cue.enabled !== false).length;
  const filterRole = current.roles?.find(role => role.id === cueRoleFilter);
  const reorderLocked = cueRoleFilter !== "all" || Boolean(cueSearchQuery.trim());
  const reorderState = reorderLocked
    ? 'class="locked" aria-disabled="true" title="请先清除角色或关键词筛选，再调整完整剧本顺序"'
    : "";
  const scopeText = filterRole ? ` · 正在查看 ${filterRole.name}` : "";
  const resultText = entries.length !== cues.length || cueSearchQuery ? ` · 显示 ${entries.length} 条` : "";
  $("#scriptSummary").textContent = `${enabledCount} 条启用台词 · ${current.roles?.length || 0} 个角色${scopeText}${resultText}`;
  setHidden("#emptyCueAction", cues.length > 0);
  renderCueFilters();
  renderTrackNotice();
  $("#cues").innerHTML = entries.length ? entries.map(({cue, index}) => {
    const role = current.roles?.find(item => item.id === cue.role) || current.roles?.[0] || {};
    const fixed = (current.settings?.timeline_mode || "fixed") === "fixed";
    const previewLocked = role.mode === "clone" && !cloneTranscriptUsable(role);
    const adopted = Boolean(cue.adopted_take);
    const blocked = current.blockers?.find(item => Number(item.cue_id) === Number(cue.id));
    const flagged = current.review_queue?.find(item => Number(item.cue_id) === Number(cue.id));
    return `<article class="cue-strip ${cue.enabled === false ? "disabled" : ""} ${adopted ? "take-adopted" : ""} ${blocked && !adopted ? "cue-blocked" : ""} ${flagged && !adopted && !blocked ? "cue-flagged" : ""} ${selectedCues.has(Number(cue.id)) ? "batch-selected" : ""}" data-cue-id="${esc(cue.id)}" onclick="if(!event.target.closest('button,input,textarea,select,label,a')) toggleCueSelection(${cue.id})">
      <div class="cue-number" onclick="event.stopPropagation(); toggleCueSelection(${cue.id})" role="button" tabindex="0" aria-label="选择第 ${index + 1} 条台词">
        <b>${String(index + 1).padStart(2, "0")}</b>
        <div class="cue-reorder">
          <button type="button" onclick="moveCue(${index},-1)" aria-label="上移台词" ${reorderState}>${icon("up")}</button>
          <button type="button" onclick="moveCue(${index},1)" aria-label="下移台词" ${reorderState}>${icon("down")}</button>
        </div>
      </div>
      <div class="cue-role">
        <select aria-label="台词角色" onchange="setCue(${index},'role',this.value)">${roleOptions(cue.role)}</select>
        <span class="cue-mode">${icon(roleModeIcon(role))}${esc(roleModeLabel(role))}</span>
      </div>
      <div class="cue-content">
        <textarea aria-label="台词内容" placeholder="输入角色要说的台词" onchange="setCue(${index},'text',this.value)">${esc(cue.text)}</textarea>
        <div class="cue-performance">
          <label>表演<input class="emotion-input" value="${esc(cue.emotion || "自然")}" onchange="setCue(${index},'emotion',this.value)"></label>
          <label>语速<input type="number" min="0.5" max="2" step="0.05" value="${Number(cue.speed || 1).toFixed(2)}" onchange="setCue(${index},'speed',+this.value)"></label>
          <button class="pronunciation-cue-button" type="button" onclick="setCuePronunciation(${index})" title="为本句中的多音字、专有名词指定读音">发音${cuePronunciationSummary(cue) ? ` · ${esc(cuePronunciationSummary(cue))}` : ''}</button>
          ${cue.source_text ? `<span class="cue-source" title="${esc(cue.source_text)}">原文：${esc(cue.source_text)}</span>` : ""}
        </div>
      </div>
      <div class="cue-control">
        <div class="cue-audio-actions">
          <button class="button quiet small preview-cue" type="button" onclick="previewCue(${index})" ${previewLocked ? "disabled title=\"请先确认该角色的参考逐字稿\"" : ""}>${icon("play")}${adopted ? "试听新版本" : blocked ? "重生成此句" : flagged ? "复听确认" : "试听此句"}</button>
          <button class="button quiet small export-cue" type="button" onclick="downloadCue(${index})" title="导出整轨中实际使用的此句 WAV">${icon("download")}导出此句</button>
        </div>
        ${adopted ? `<div class="take-status ${cue.adopted_take?.reviewed ? "take-reviewed" : ""}" role="status"><span>${icon("check")}${cue.adopted_take?.reviewed ? "人工复听确认，整轨复用" : "整轨复用已采用版本"}</span><div><button class="take-link" type="button" onclick="playAdoptedTake(${index})">播放</button><button class="take-link danger" type="button" onclick="discardAdoptedTake(${index})">取消采用</button></div></div>` : ""}
        ${blocked && !adopted ? `<div class="take-status take-blocked" role="status"><span>${icon("alert")}这句没有生成出音频，整轨缺这一句</span><span class="blocked-reason">${esc(blocked.error)}</span><span>点“重生成此句”再试一次。</span></div>` : ""}
        ${flagged && !adopted && !blocked ? `<div class="take-status take-flagged" role="status"><span>${icon("alert")}自动校验未通过，已用最佳候选发布</span><span class="blocked-reason">识别成“${esc(flagged.transcript || "—")}”</span><span>听着没问题就忽略；若不对，用“复听确认”换用你听过的版本。</span></div>` : ""}
        <div class="timing">
          ${fixed ? `<input type="number" step=".001" value="${cue.start}" aria-label="开始时间" onchange="setCue(${index},'start',+this.value)"><input type="number" step=".001" value="${cue.end}" aria-label="结束时间" onchange="setCue(${index},'end',+this.value)">` : `<input type="number" min="0" step=".05" value="${Number(cue.gap_after ?? .25).toFixed(2)}" aria-label="句后间隔" onchange="setCue(${index},'gap_after',+this.value)" title="句后间隔（秒）">`}
          <button class="icon-button danger" type="button" onclick="removeCue(${index})" aria-label="删除台词">${icon("trash")}</button>
        </div>
      </div>
    </article>`;
  }).join("") : `<div class="cue-filter-empty"><strong>没有符合条件的台词</strong><span>${cueSearchQuery ? "换个关键词，或清除当前筛选。" : "这个角色还没有台词，可添加新台词或切换角色。"}</span><button class="button quiet small" type="button" onclick="clearCueFilters()">查看全部台词</button></div>`;
  updateBatchButton();
  if (outputAudio && !outputAudio.paused) requestAnimationFrame(() => updatePlaybackFollow(outputAudio.currentTime));
}

function setCue(index, key, value) {
  current.cues[index][key] = value;
  if (key === "role") {
    renderCues();
    renderImportedVoiceGuide();
  }
}

function addCue() {
  if (!current.roles?.length) addRole();
  const last = current.cues?.at(-1);
  const start = last ? Number(last.end || 0) + Number(last.gap_after || 0) : 0;
  const roleId = cueRoleFilter !== "all" && current.roles.some(role => role.id === cueRoleFilter)
    ? cueRoleFilter
    : current.roles[0]?.id || "";
  const nextId = Math.max(0, ...(current.cues || []).map(cue => Number(cue.id) || 0)) + 1;
  current.cues.push({
    id: nextId,
    start,
    end: start + 3,
    source_text: "",
    text: "",
    role: roleId,
    emotion: "自然",
    speed: 1,
    gap_after: Number(current.settings?.default_gap ?? .25),
    enabled: true,
  });
  cueSearchQuery = "";
  renderCues();
  renderImportedVoiceGuide();
  setTimeout(() => {
    const textarea = document.querySelector(`[data-cue-id="${nextId}"] textarea`);
    textarea?.focus();
    textarea?.scrollIntoView({behavior: "smooth", block: "nearest"});
  }, 20);
}

function removeCue(index) {
  current.cues.splice(index, 1);
  renderCues();
  renderImportedVoiceGuide();
}

function moveCue(index, direction) {
  if (cueRoleFilter !== "all" || cueSearchQuery.trim()) {
    showToast("请先清除角色或关键词筛选，再调整完整剧本顺序");
    return;
  }
  const target = index + direction;
  if (target < 0 || target >= current.cues.length) return;
  [current.cues[index], current.cues[target]] = [current.cues[target], current.cues[index]];
  renderCues();
}

function renderRoles() {
  const roles = current.roles || [];
  if (!roles.some(role => role.id === selectedRoleId)) selectedRoleId = roles[0]?.id || null;
  $("#roleCount").textContent = roles.length;
  $("#roleTabs").innerHTML = roles.map(role => {
    const state = importedRoleIds().includes(role.id) ? importedRoleState(role) : null;
    return `<button class="role-tab ${role.mode === "clone" ? "clone" : ""} ${state ? `status-${state.key}` : ""} ${role.id === selectedRoleId ? "active" : ""}" type="button" onclick="selectRole('${role.id}')"><i></i><span>${esc(role.name)}</span>${state ? `<small aria-label="${esc(state.label)}">${state.key === "ready" ? icon("check") : state.key === "design" ? icon("spark") : icon("alert")}</small>` : ""}</button>`;
  }).join("");
  const role = activeRole();
  if (!role) {
    $("#roleInspector").innerHTML = `<div class="inspector-empty">添加一个角色，为其设计新声音或绑定获得授权的参考音频。</div>`;
    return;
  }
  const roleIndex = roles.findIndex(item => item.id === role.id);
  const clone = role.mode === "clone";
  const previewKey = `${current.id}:${role.id}`;
  const previewJob = lastJobs[`preview-${role.id}`];
  const previewQueued = previewJob?.status === "queued";
  const previewRunning = previewJobKeys.has(previewKey) || previewJob?.status === "running";
  const cloneReady = !clone || cloneTranscriptUsable(role);
  const previewDisabled = previewRunning || previewQueued || !cloneReady;
  $("#roleInspector").innerHTML = `
    <label class="field"><span>角色名称</span><input value="${esc(role.name)}" onchange="setRole(${roleIndex},'name',this.value,true)"></label>
    <div class="mode-switch" role="group" aria-label="声线来源">
      <button class="${clone ? "" : "active"}" type="button" onclick="setRoleMode(${roleIndex},'design')">${icon("spark")}场景造声</button>
      <button class="clone ${clone ? "active" : ""}" type="button" onclick="setRoleMode(${roleIndex},'clone')">${icon("clone")}授权克隆</button>
    </div>
    ${clone ? renderCloneFields(role, roleIndex) : renderDesignFields(role, roleIndex)}
    ${renderPersonalityFields(role, roleIndex)}
    <div class="preview-block">
      <label class="field"><span>声线试听台词</span><textarea id="rolePreviewText" placeholder="输入一句用于确认声线的台词">${esc(previewTextFor(role.id))}</textarea></label>
      ${clone && !cloneReady ? `<p class="preview-lock">请先播放参考音频，校正并确认逐字稿，才能生成试听。</p>` : ""}
      <div class="preview-actions"><button id="rolePreviewButton" class="button primary" type="button" onclick="previewRole()" ${previewDisabled ? "disabled" : ""}>${previewRunning ? `${icon("wave")}生成试听中…` : previewQueued ? `${icon("wave")}排队中…` : `${icon("play")}生成试听`}</button><button id="adoptPreviewButton" class="button quiet" type="button" onclick="adoptPreviewTake()" hidden disabled>${icon("check")}采用到整轨</button></div>
      <p id="previewTakeNote" class="preview-take-note" hidden></p>
      <audio id="rolePreviewAudio" class="role-preview-audio" controls hidden></audio>
    </div>
    ${roles.length > 1 ? `<button class="button quiet danger role-danger" type="button" onclick="removeRole(${roleIndex})">${icon("trash")}删除这个角色</button>` : ""}
  `;
}

function renderDesignFields(role, index) {
  return `
    <div class="field"><span>快速起稿</span><div class="preset-grid">${Object.entries(localVoiceOptions).map(([key, label]) => `<button class="${role.voice === key ? "active" : ""}" type="button" onclick="applyVoicePreset(${index},'${key}')">${esc(label)}</button>`).join("")}</div></div>
    <label class="field"><span>人物与场景声音描述</span><textarea onchange="setRole(${index},'description',this.value)">${esc(role.description || "")}</textarea><small class="field-note">写年龄、音高、音色、口音与场景。下方“人物性格”单独决定这个角色遇事时怎样表达情绪。</small></label>
  `;
}

function renderPersonalityFields(role, index) {
  const hasPersonality = Boolean(String(role.personality || "").trim());
  const cloned = role.mode === "clone";
  return `
    <section class="personality-block ${cloned ? "clone" : "design"}" aria-label="人物性格与情绪反应">
      <div class="personality-heading"><span>人物性格</span><div><button class="role-cue-link" type="button" onclick="showRoleCues('${role.id}')">查看 ${cueCountForRole(role.id)} 条台词</button><small>稳定表演底色</small></div></div>
      <textarea maxlength="500" placeholder="例如：外冷内热、责任感强；遇到危险先压住慌张再下判断，生气时不吼，音量更低、咬字更重；安慰熟人时嘴硬但语速会放慢。" oninput="setRolePersonality(${index},this.value,'${role.id}')">${esc(role.personality || "")}</textarea>
      <small class="field-note">写“平时是什么人 + 紧张、生气、开心、安慰别人时怎么说”。系统会把它转成每句的停顿、气息、重音和情绪外露程度，不会把性格原文塞进台词。</small>
      <div class="personality-actions">
        <button id="rolePersonalityAction-${esc(role.id)}" class="button quiet small" type="button" onclick="directRoleEmotions('${role.id}')" ${hasPersonality ? "" : "disabled"}>${icon("spark")}按性格重算表演</button>
        <span>${cloned ? "克隆模型保留原声线；性格主要用于逐句情绪导演，效果受参考音频表达范围影响。" : "场景造声会同时使用人物性格与当前句表演。"}</span>
      </div>
    </section>
  `;
}

function cloneTranscriptReady(role) {
  return Boolean(role?.clone_consent && role?.ref_audio && role?.ref_text?.trim() && role?.reference_revision && role?.ref_text_confirmed && role.ref_text_confirmed_revision === role.reference_revision);
}

function cloneTranscriptUsable(role) {
  return cloneTranscriptReady(role) && cloneTranscriptDraft(role).trim() === String(role.ref_text || "").trim();
}

function transcriptDraftKey(role) {
  return `${current?.id || ""}:${role.id}:${role.reference_revision || "none"}`;
}

function cloneTranscriptDraft(role) {
  const key = transcriptDraftKey(role);
  return referenceTranscriptDrafts.has(key) ? referenceTranscriptDrafts.get(key) : (role.ref_text || role.ref_text_auto || "");
}

function renderCloneFields(role, index) {
  const hasReference = Boolean(role.ref_audio);
  const confirmed = cloneTranscriptReady(role);
  const draft = cloneTranscriptDraft(role);
  const changed = hasReference && draft.trim() !== String(role.ref_text || "").trim();
  const confirming = transcriptConfirmKeys.has(transcriptDraftKey(role));
  const statusClass = confirmed && !changed ? "confirmed" : "pending";
  const statusText = changed ? "逐字稿已修改，等待重新确认" : confirmed ? "逐字稿已确认，可用于克隆" : "自动识别草稿，等待核听确认";
  return `
    <label class="consent-box"><input type="checkbox" ${role.clone_consent ? "checked" : ""} onchange="setRole(${index},'clone_consent',this.checked)"><span>我确认已获得声音本人授权或拥有合法使用权，并对生成内容负责。</span></label>
    <div class="field"><span>参考音频</span>
      ${hasReference ? `<div class="reference-ready"><span><b>${esc(role.ref_name || "已上传参考音频")}</b><small class="${statusClass}">${esc(statusText)}</small></span><div class="reference-actions"><label class="icon-button" title="替换参考音频"><input type="file" accept="audio/*" onchange="uploadReference('${role.id}',this.files[0])">${icon("upload")}</label><button class="icon-button danger" type="button" onclick="removeReference('${role.id}')" aria-label="移除参考音频">${icon("trash")}</button></div></div><audio class="reference-audio" controls preload="metadata" src="/api/projects/${current.id}/roles/${role.id}/reference?v=${encodeURIComponent(role.reference_revision || "")}"></audio>` : `<label class="reference-drop"><input type="file" accept="audio/*" onchange="uploadReference('${role.id}',this.files[0])">${icon("upload")}<b>上传 5–15 秒清晰人声</b><span>WAV、MP3、M4A、FLAC 等</span></label>`}
    </div>
    ${hasReference ? `<label class="field transcript-field"><span>参考音频逐字稿</span><textarea id="referenceTranscript" maxlength="1000" placeholder="请播放上方音频，按实际说出的原话逐字校正" oninput="updateReferenceTranscriptDraft('${role.id}',this.value)">${esc(draft)}</textarea><small class="field-note">自动转写只是一份草稿。请先播放参考音频，修正人名、口音、方言和同音词，再明确确认。模型只会使用确认后的逐字稿。</small>${role.ref_text_warning ? `<small class="transcript-warning">上传前填写的内容与自动稿差异较大，请务必核听。</small>` : ""}<div class="transcript-confirm-row"><span class="transcript-status ${statusClass}" role="status">${esc(statusText)}</span><button id="confirmTranscriptButton" class="button ${confirmed && !changed ? "quiet" : "primary"} small" type="button" onclick="confirmReferenceTranscript('${role.id}')" ${confirming || !draft.trim() ? "disabled" : ""}>${confirming ? `${icon("wave")}确认中…` : `${icon("save")}${confirmed && !changed ? "重新确认" : "确认逐字稿"}`}</button></div></label>` : `<div class="field"><span>逐字稿确认</span><small class="field-note">上传后系统会给出自动识别草稿；你需要播放参考音频、人工校正并确认，之后才能试听或生成整轨。</small></div>`}
  `;
}

function previewTextFor(roleId) {
  return current.cues?.find(cue => cue.role === roleId && cue.text)?.text || "这是这位角色的声音测试，请确认音色和表演是否合适。";
}

function selectRole(roleId) {
  selectedRoleId = roleId;
  previewContext = null;
  renderRoles();
  renderImportedVoiceGuide();
}

function setRole(index, key, value, rerender = false) {
  current.roles[index][key] = value;
  if (key === "description") current.roles[index].style = value;
  if (rerender) {
    renderRoles();
    renderCues();
  }
}

function setRolePersonality(index, value, roleId) {
  setRole(index, "personality", value);
  const action = document.getElementById(`rolePersonalityAction-${roleId}`);
  if (action) action.disabled = !value.trim();
}

async function directRoleEmotions(roleId) {
  const role = current.roles.find(item => item.id === roleId);
  if (!role?.personality?.trim()) return showToast("请先填写人物性格", true);
  try {
    await saveProject(true);
    showToast(`正在按“${role.name || "未命名"}”的性格重算逐句表演……`);
    current = await api(`/api/projects/${current.id}/direct-role-emotions`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({role_id: roleId}),
    });
    selectedRoleId = roleId;
    render();
    await refreshProjects();
    showToast(`已按人物性格更新 ${current.cues.filter(cue => cue.role === roleId && cue.enabled !== false).length} 条表演`);
  } catch (error) {
    showToast(error.message, true);
  }
}

function setRoleMode(index, mode) {
  current.roles[index].mode = mode;
  renderRoles();
  renderCues();
  renderImportedVoiceGuide();
}

function applyVoicePreset(index, key) {
  const role = current.roles[index];
  role.voice = key;
  role.description = localVoiceDescriptions[key];
  role.style = role.description;
  renderRoles();
}

function addRole() {
  const number = (current.roles?.length || 0) + 1;
  const id = `role${Date.now()}`;
  current.roles.push({
    id,
    name: `角色${number}`,
    voice: "male_adult",
    mode: "design",
    description: localVoiceDescriptions.male_adult,
    style: localVoiceDescriptions.male_adult,
    personality: "",
    ref_audio: "",
    ref_text: "",
    ref_text_auto: "",
    ref_text_confirmed: false,
    ref_text_confirmed_revision: "",
    clone_consent: false,
  });
  selectedRoleId = id;
  renderRoles();
  renderCues();
  renderImportedVoiceGuide();
}

async function removeRole(index) {
  const role = current.roles[index];
  if (!role || current.roles.length <= 1) return;
  try {
    await saveProject(true);
    const latestIndex = current.roles.findIndex(item => item.id === role.id);
    if (latestIndex < 0) throw new Error("角色已变化，请重新选择后删除");
    current = await api(`/api/projects/${current.id}/roles/${role.id}`, {method: "DELETE"});
    if (cueRoleFilter === role.id) cueRoleFilter = "all";
    selectedRoleId = current.roles[0]?.id || null;
    render();
    await refreshProjects();
    showToast(`角色“${role.name}”已删除，关联授权音频和试听文件已清理`);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function uploadReference(roleId, file) {
  if (!file) return;
  const role = current.roles.find(item => item.id === roleId);
  if (file.size > 64 * 1024 * 1024) {
    showToast("参考音频超过 64 MB，请先裁剪为 5–15 秒的清晰单人语音", true);
    return;
  }
  if (!role?.clone_consent) {
    showToast("请先确认已获得声音授权", true);
    renderRoles();
    return;
  }
  const formData = new FormData();
  formData.append("audio", file);
  formData.append("consent", "true");
  try {
    showToast("正在处理参考音频……");
    current = await api(`/api/projects/${current.id}/roles/${roleId}/reference`, {method: "POST", body: formData});
    selectedRoleId = roleId;
    render();
    renderImportedVoiceGuide();
    showToast(current.roles.find(item => item.id === roleId)?.ref_text_warning
      ? "参考音频已绑定；自动稿与原填写差异较大，请播放核听后确认"
      : "参考音频已绑定；请播放核听、校正并确认自动逐字稿");
  } catch (error) {
    showToast(error.message, true);
  }
}

async function removeReference(roleId) {
  try {
    current = await api(`/api/projects/${current.id}/roles/${roleId}/reference`, {method: "DELETE"});
    selectedRoleId = roleId;
    for (const key of referenceTranscriptDrafts.keys()) {
      if (key.startsWith(`${current.id}:${roleId}:`)) referenceTranscriptDrafts.delete(key);
    }
    render();
    renderImportedVoiceGuide();
    showToast("参考音频已移除，角色已切回场景造声");
  } catch (error) {
    showToast(error.message, true);
  }
}

function updateReferenceTranscriptDraft(roleId, value) {
  const role = current.roles.find(item => item.id === roleId);
  if (!role) return;
  referenceTranscriptDrafts.set(transcriptDraftKey(role), value);
  const changed = value.trim() !== String(role.ref_text || "").trim();
  const usable = cloneTranscriptReady(role) && !changed;
  const statusText = changed ? "逐字稿已修改，等待重新确认" : usable ? "逐字稿已确认，可用于克隆" : "自动识别草稿，等待核听确认";
  const readyStatus = document.querySelector(".reference-ready small");
  if (readyStatus) {
    readyStatus.className = usable ? "confirmed" : "pending";
    readyStatus.textContent = statusText;
  }
  const transcriptStatus = $(".transcript-status");
  if (transcriptStatus) {
    transcriptStatus.className = `transcript-status ${usable ? "confirmed" : "pending"}`;
    transcriptStatus.textContent = statusText;
  }
  const button = $("#confirmTranscriptButton");
  if (button) {
    button.disabled = !value.trim();
    button.className = `button ${usable ? "quiet" : "primary"} small`;
    button.innerHTML = `${icon("save")}${usable ? "重新确认" : "确认逐字稿"}`;
  }
  const previewButton = $("#rolePreviewButton");
  if (previewButton) previewButton.disabled = !usable;
  renderCues();
  renderImportedVoiceGuide();
  renderStatus();
}

async function confirmReferenceTranscript(roleId) {
  const role = current.roles.find(item => item.id === roleId);
  if (!role?.reference_revision) return showToast("参考音频状态无效，请重新上传", true);
  const key = transcriptDraftKey(role);
  const transcript = String(referenceTranscriptDrafts.get(key) ?? role.ref_text ?? "").trim();
  if (!transcript) return showToast("请先填写参考音频中实际说出的逐字稿", true);
  if (transcript.length > 1000) return showToast("参考逐字稿超过 1000 字，请只填写参考音频中实际说出的内容", true);
  if (transcriptConfirmKeys.has(key)) return;
  transcriptConfirmKeys.add(key);
  renderRoles();
  try {
    current = await api(`/api/projects/${current.id}/roles/${roleId}/reference-transcript`, {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ref_text: transcript, reference_revision: role.reference_revision}),
    });
    referenceTranscriptDrafts.delete(key);
    selectedRoleId = roleId;
    render();
    await refreshProjects();
    const remaining = importedRoleEntries().filter(({role: item}) => ["missing", "pending"].includes(importedRoleState(item).key));
    showToast(remaining.length
      ? `角色“${role.name || "未命名"}”克隆已就绪；还可继续配置 ${remaining.length} 个角色`
      : "逐字稿已确认，现在可以生成试听和整轨");
  } catch (error) {
    showToast(error.message, true);
    if (error.status === 409) await loadProject(current.id);
  } finally {
    transcriptConfirmKeys.delete(key);
    if (current?.id) {
      renderRoles();
      renderImportedVoiceGuide();
    }
  }
}

async function saveProject(silent = false) {
  if (!current) return null;
  try {
    current = await api(`/api/projects/${current.id}`, {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(current),
    });
    render();
    await refreshProjects();
    if (!silent) showToast("项目已保存");
    return current;
  } catch (error) {
    showToast(error.message, true);
    throw error;
  }
}

async function renameProject(value) {
  current.name = value.trim() || current.name;
  await saveProject(true);
}

async function setProjectDirection(value) {
  const next = value.trim().slice(0, 160);
  if (next === (current.direction_brief || "")) return;
  current.direction_brief = next;
  await saveProject(true);
  showToast(next ? "配音方向已保存，重新生成后生效" : "配音方向已清除，重新生成后生效");
}

async function generateProject() {
  const pending = pendingCloneRoles();
  if (pending.length) return showToast(`请先播放并确认角色“${pending[0].name || "未命名"}”的参考逐字稿`, true);
  try {
    await saveProject(true);
    const stage = current.project_type === "video" ? "retry" : "tts";
    await api(`/api/projects/${current.id}/${stage}`, {method: "POST"});
    showToast(current.project_type === "video" ? "成片重新生成已开始" : "整轨生成已开始，可以继续查看项目状态");
    poll();
  } catch (error) {
    showToast(error.message, true);
  }
}

async function previewCue(index) {
  const cue = current.cues[index];
  if (!cue.text?.trim()) return showToast("请先填写这句台词", true);
  selectedRoleId = cue.role;
  previewContext = {cueId: cue.id, cueIndex: index, text: cue.text.trim(), emotion: cue.emotion || "自然", file: ""};
  renderRoles();
  $("#rolePreviewText").value = cue.text;
  await previewRole(cue.emotion || "自然", cue.id);
}

async function previewRole(emotion = "自然", cueId = null) {
  const role = activeRole();
  const text = $("#rolePreviewText")?.value.trim();
  if (!role || !text) return showToast("请输入试听台词", true);
  if (role.mode === "clone" && !cloneTranscriptUsable(role)) return showToast("请先播放参考音频，校正并确认逐字稿", true);
  const projectId = current.id;
  const previewKey = `${projectId}:${role.id}`;
  if (previewJobKeys.has(previewKey)) return;
  const matchedCueId = cueId ?? ((previewContext?.cueId && previewContext.text === text) ? previewContext.cueId : null);
    previewContext = matchedCueId ? {cueId: matchedCueId, cueIndex: current.cues.findIndex(cue => cue.id === matchedCueId), text, emotion, file: ""} : null;
    const previewNonce = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
    previewJobKeys.add(previewKey);
    renderRoles();
    try {
      await saveProject(true);
      if (matchedCueId) {
        const savedCue = current.cues.find(cue => cue.id === matchedCueId);
        if (!savedCue || savedCue.role !== role.id || String(savedCue.text || "").trim() !== text || String(savedCue.emotion || "自然").trim() !== emotion) {
          throw new Error("台词已变化，请从台词行重新生成试听");
        }
      }
    await api(`/api/projects/${projectId}/preview-voice`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({role_id: role.id, text, emotion, personality_directed: true, cue_id: matchedCueId, preview_nonce: previewNonce}),
    });
    pollPreview(projectId, role.id, matchedCueId);
  } catch (error) {
    if (error.status === 409) {
      pollPreview(projectId, role.id, matchedCueId);
      return;
    }
    previewJobKeys.delete(previewKey);
    showToast(error.message, true);
    renderRoles();
  }
}

async function pollPreview(projectId, roleId, cueId = null) {
  const previewKey = `${projectId}:${roleId}`;
  try {
    const jobs = await api(`/api/projects/${projectId}/jobs`);
    const job = jobs[`preview-${roleId}`];
    if (job?.status === "queued") {
      renderRoles();
      setTimeout(() => pollPreview(projectId, roleId, cueId), 1200);
      return;
    }
    if (job?.status === "failed") {
      previewJobKeys.delete(previewKey);
      showToast(job.message, true);
      renderRoles();
      return;
    }
    if (job?.status === "done") {
      previewJobKeys.delete(previewKey);
      if (current?.id === projectId && selectedRoleId === roleId) {
        const audio = $("#rolePreviewAudio");
        if (audio) {
          const file = job.result || "";
          audio.src = `/api/projects/${projectId}/preview-voice/${roleId}?v=${Date.now()}${file ? `&file=${encodeURIComponent(file)}` : ""}`;
          setHidden(audio, false);
          audio.play().catch(() => {});
          if (cueId && previewContext?.cueId === cueId) {
            previewContext.file = file;
            const gate = file
              ? await api(`/api/projects/${projectId}/preview-voice/${roleId}/gate?file=${encodeURIComponent(file)}`).catch(() => ({gate_passed: true}))
              : {gate_passed: true};
            previewContext.gateRejected = gate?.gate_passed === false;
            previewContext.gateTranscript = gate?.gate_transcript || "";
            const adopt = $("#adoptPreviewButton");
            const note = $("#previewTakeNote");
            if (adopt) {
              adopt.hidden = false;
              adopt.disabled = !file;
              adopt.innerHTML = previewContext.gateRejected ? `${icon("check")}确认无误，采用到整轨` : `${icon("check")}采用到整轨`;
            }
            if (note) {
              note.hidden = false;
              note.className = previewContext.gateRejected ? "preview-take-note gate-rejected" : "preview-take-note";
              note.textContent = !file
                ? "试听完成，但没有取得可采用的文件，请重新生成。"
                : previewContext.gateRejected
                  ? `自动校验没通过（识别成“${previewContext.gateTranscript}”）。请完整听完，确认没问题再点“确认无误，采用到整轨”。`
                  : `这是台词 ${String(cueId).padStart(2, "0")} 的候选版本。采用后，整轨会直接复用这段声音。`;
            }
          }
        }
        const button = $("#rolePreviewButton");
        if (button) {
          button.disabled = false;
          button.innerHTML = `${icon("play")}重新生成试听`;
        }
      }
      return;
    }
    setTimeout(() => pollPreview(projectId, roleId, cueId), 1200);
  } catch (error) {
    setTimeout(() => pollPreview(projectId, roleId, cueId), 1600);
  }
}

async function refreshProjectView() {
  if (!current?.id) return;
  current = await api(`/api/projects/${current.id}`);
  render();
}

async function adoptPreviewTake() {
  const context = previewContext;
  if (!context?.cueId || !context.file) return showToast("请先从台词行生成一个候选试听", true);
  const button = $("#adoptPreviewButton");
  if (button) button.disabled = true;
  try {
    await saveProject(true);
    const savedCue = current.cues.find(cue => cue.id === context.cueId);
    if (!savedCue || String(savedCue.text || "").trim() !== context.text || String(savedCue.emotion || "自然").trim() !== context.emotion) {
      throw new Error("台词已变化，请重新试听后再采用");
    }
    current = await api(`/api/projects/${current.id}/cues/${context.cueId}/adopt-take`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        file: context.file,
        confirm_gate_rejection: Boolean(context.gateRejected),
        review_note: context.gateRejected ? `复听确认；自动识别为“${context.gateTranscript || ""}”` : "",
      }),
    });
    const confirmedReview = Boolean(context.gateRejected);
    previewContext = null;
    render();
    // The mutation response carries no blocker list; refresh from the canonical read.
    await refreshProjectView();
    showToast(confirmedReview
      ? `已人工复听确认台词 ${String(context.cueId).padStart(2, "0")}，整轨会直接复用`
      : `已采用台词 ${String(context.cueId).padStart(2, "0")} 的试听版本，整轨会直接复用`);
  } catch (error) {
    if (button) button.disabled = false;
    showToast(error.message, true);
  }
}

function playAdoptedTake(index) {
  const cue = current.cues[index];
  if (!cue?.adopted_take) return;
  const audio = new Audio(`/api/projects/${current.id}/cues/${cue.id}/adopted-take?v=${Date.now()}`);
  audio.play().catch(() => showToast("已采用版本播放失败，请重新试听", true));
}

async function discardAdoptedTake(index) {
  const cue = current.cues[index];
  if (!cue?.adopted_take) return;
  try {
    current = await api(`/api/projects/${current.id}/cues/${cue.id}/adopt-take`, {method: "DELETE"});
    render();
    await refreshProjectView();
    showToast(`已取消台词 ${String(cue.id).padStart(2, "0")} 的采用版本`);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function retryAuto() {
  try {
    await api(`/api/projects/${current.id}/retry`, {method: "POST"});
    poll();
  } catch (error) {
    showToast(error.message, true);
  }
}

function renderSettings() {
  const settings = current.settings || {};
  const video = (current.project_type || "video") === "video";
  $("#settings").innerHTML = `
    <label>配音语言<select onchange="current.settings.target_lang=this.value"><option value="zh" ${settings.target_lang === "zh" ? "selected" : ""}>中文</option><option value="en" ${settings.target_lang === "en" ? "selected" : ""}>英语</option><option value="ja" ${settings.target_lang === "ja" ? "selected" : ""}>日语</option><option value="ko" ${settings.target_lang === "ko" ? "selected" : ""}>韩语</option></select></label>
    <label>时间轴模式<select onchange="current.settings.timeline_mode=this.value;renderCues()"><option value="natural" ${settings.timeline_mode === "natural" ? "selected" : ""}>自然顺排</option><option value="fixed" ${settings.timeline_mode === "fixed" ? "selected" : ""}>固定时间槽</option></select></label>
    <label>默认句间停顿（秒）<input type="number" min="0" max="10" step=".05" value="${settings.default_gap ?? .25}" onchange="current.settings.default_gap=+this.value"></label>
    <div class="settings-section pronunciation-settings"><h3>多音字与专有名词</h3>
      <label>发音词典<textarea id="pronunciationLexicon" rows="6" placeholder="每行一条，例如：&#10;重庆 → 崇庆&#10;角色 → 绝色&#10;QF1 → Q F one">${esc((settings.pronunciation_lexicon || []).map(item => `${item.source} → ${item.spoken}`).join("\n"))}</textarea></label>
      <small class="field-note">左侧是台词原文，右侧是交给模型朗读的形式；优先匹配较长词语，不修改显示文本。</small>
      <button class="button quiet wide" type="button" onclick="savePronunciationLexicon()">${icon("save")}保存发音词典</button>
    </div>

    ${video ? `<div class="settings-section"><h3>视频字幕与混音</h3><div class="settings-grid">
      <label>字幕 Y<input type="number" value="${settings.cover_y}" onchange="current.settings.cover_y=+this.value"></label>
      <label>字幕高度<input type="number" value="${settings.cover_h}" onchange="current.settings.cover_h=+this.value"></label>
      <label>字幕字号<input type="number" value="${settings.subtitle_font_size}" onchange="current.settings.subtitle_font_size=+this.value"></label>
      <label>配音增益<input type="number" step=".05" value="${settings.voice_gain}" onchange="current.settings.voice_gain=+this.value"></label>
      <label>原声增益<input type="number" step=".05" value="${settings.source_gain}" onchange="current.settings.source_gain=+this.value"></label>
      <label>压低阈值<input type="number" step=".001" value="${settings.duck_threshold}" onchange="current.settings.duck_threshold=+this.value"></label>
      <label class="toggle-field"><input type="checkbox" ${settings.keep_source_audio ? "checked" : ""} onchange="current.settings.keep_source_audio=this.checked"><span>保留并自动压低视频原声</span></label>
    </div></div>` : ""}
    <button class="button primary wide" type="button" onclick="saveProject();toggleProjectPanel(false)">${icon("save")}保存设置</button>
  `;
}

function toggleProjectPanel(force) {
  const panel = $("#projectPanel");
  if (!panel) return;
  const show = force ?? panel.hidden;
  panel.hidden = !show;
  if (show && !current) {
    panel.hidden = true;
    showToast("请先打开一个项目", true);
  }
}

function renderTrackNotice() {
  const node = $("#trackNotice");
  if (!node) return;
  const blocked = current?.blockers?.length || 0;
  const flagged = (current?.review_queue || []).length;
  if (!blocked && !flagged) {
    node.hidden = true;
    node.innerHTML = "";
    return;
  }
  const parts = [];
  if (blocked) parts.push(`<span class="track-notice-item danger">${icon("alert")}${blocked} 句没有生成出音频，整轨里是空的，点“重生成此句”重试</span>`);
  if (flagged) parts.push(`<span class="track-notice-item">${icon("alert")}${flagged} 句自动校验未通过，整轨已用最佳候选发布，建议复听这几句</span>`);
  node.hidden = false;
  node.innerHTML = parts.join("");
}

async function poll() {
  if (!current) return;
  clearTimeout(timer);
  try {
    const [project, jobs, log] = await Promise.all([
      api(`/api/projects/${current.id}`),
      api(`/api/projects/${current.id}/jobs`),
      api(`/api/projects/${current.id}/log`),
    ]);
    current = project;
    lastJobs = jobs;
    $("#log").textContent = log.log || "尚无日志";
    render();
    await refreshProjects();
    const active = Object.values(jobs).some(job => ["queued", "running"].includes(job.status)) || ["queued", "running"].includes(current.pipeline?.status);
    if (active) timer = setTimeout(poll, 1800);
  } catch (error) {
    timer = setTimeout(poll, 2800);
  }
}

function downloadOutput() {
  if (!current) return;
  location.href = `/api/projects/${current.id}/download`;
}

async function downloadCue(index) {
  const cue = current?.cues?.[index];
  if (!cue) return showToast("台词不存在", true);
  let stored;
  try {
    stored = await api(`/api/projects/${current.id}`);
  } catch (error) {
    return showToast(error.message, true);
  }
  const savedCue = stored.cues?.find(item => Number(item.id) === Number(cue.id));
  const localState = JSON.stringify({
    role: cue.role,
    text: String(cue.text || "").trim(),
    emotion: String(cue.emotion || "").trim(),
    speed: Number(cue.speed || 1),
    start: Number(cue.start || 0),
    end: Number(cue.end || 0),
    gap_after: Number(cue.gap_after || 0),
    enabled: cue.enabled !== false,
  });
  const savedState = savedCue && JSON.stringify({
    role: savedCue.role,
    text: String(savedCue.text || "").trim(),
    emotion: String(savedCue.emotion || "").trim(),
    speed: Number(savedCue.speed || 1),
    start: Number(savedCue.start || 0),
    end: Number(savedCue.end || 0),
    gap_after: Number(savedCue.gap_after || 0),
    enabled: savedCue.enabled !== false,
  });
  if (!savedCue || localState !== savedState) return showToast("这句有未保存修改，请先保存并重新生成整轨", true);
  try {
    const response = await fetch(`/api/projects/${current.id}/cues/${encodeURIComponent(cue.id)}/download`);
    if (!response.ok) {
      const contentType = response.headers.get("content-type") || "";
      const payload = contentType.includes("application/json") ? await response.json() : await response.text();
      throw Error(payload?.error || payload?.message || payload || "单句导出失败");
    }
    const blob = await response.blob();
    const disposition = response.headers.get("content-disposition") || "";
    const encodedName = disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
    const fallbackName = `台词-${String(cue.id).padStart(3, "0")}.wav`;
    let filename = fallbackName;
    if (encodedName) {
      try { filename = decodeURIComponent(encodedName); } catch (_) { filename = fallbackName; }
    }
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    showToast(`已导出台词 ${cue.id}`);
  } catch (error) {
    showToast(error.message, true);
  }
}

function openOutputPreview() {
  if (!current?.output) return showToast("视频成片尚未完成", true);
  $("#mediaPreviewName").textContent = current.name;
  $("#videoPreview").src = `/api/projects/${current.id}/preview?v=${Date.now()}`;
  $("#mediaDialog").showModal();
}

function closeMediaDialog() {
  const video = $("#videoPreview");
  video.pause();
  video.removeAttribute("src");
  video.load();
  $("#mediaDialog").close();
}

function toggleOutputPlayback() {
  const audio = $("#outputAudio");
  if (!audio.src) return showToast("请先生成配音音轨", true);
  if (audio.paused) audio.play().catch(error => showToast(error.message, true));
  else audio.pause();
}

let followedCueId = null;
let lastFollowScrollAt = 0;

function cueAtPlaybackTime(seconds) {
  const cues = (current?.cues || []).filter(cue => cue.enabled !== false);
  if (!cues.length) return null;
  const exact = cues.find(cue => {
    const start = Number(cue.playback_start ?? cue.start);
    const end = Number(cue.playback_end ?? cue.end);
    return Number.isFinite(start) && Number.isFinite(end) && seconds >= start && seconds < end;
  });
  if (exact) return exact;
  let previous = null;
  for (const cue of cues) {
    const start = Number(cue.playback_start ?? cue.start);
    if (Number.isFinite(start) && start <= seconds) previous = cue;
    else if (Number.isFinite(start) && start > seconds) break;
  }
  return previous;
}

function updatePlaybackFollow(seconds, forceScroll = false) {
  document.querySelectorAll('.cue-strip.playing-cue').forEach(row => row.classList.remove('playing-cue'));
  const cue = cueAtPlaybackTime(seconds);
  if (!cue) {
    followedCueId = null;
    return;
  }
  const cueId = Number(cue.id);
  const row = document.querySelector(`.cue-strip[data-cue-id="${CSS.escape(String(cue.id))}"]`);
  if (!row) return;
  row.classList.add('playing-cue');
  const changed = followedCueId !== cueId;
  followedCueId = cueId;
  const now = Date.now();
  if ((changed || forceScroll) && now - lastFollowScrollAt > 350) {
    const container = document.getElementById('cues');
    if (container) {
      const rowTop = row.offsetTop;
      const rowBottom = rowTop + row.offsetHeight;
      const viewTop = container.scrollTop;
      const viewBottom = viewTop + container.clientHeight;
      const margin = 18;
      const visible = rowTop >= viewTop + margin && rowBottom <= viewBottom - margin;
      if (!visible) {
        const centeredTop = rowTop - Math.max(0, (container.clientHeight - row.offsetHeight) / 2);
        const maxTop = Math.max(0, container.scrollHeight - container.clientHeight);
        container.scrollTo({top: Math.max(0, Math.min(centeredTop, maxTop)), behavior: 'smooth'});
      }
    }
    lastFollowScrollAt = now;
  }
}

function clearPlaybackFollow() {
  followedCueId = null;
  document.querySelectorAll('.cue-strip.playing-cue').forEach(row => row.classList.remove('playing-cue'));
}

const outputAudio = $("#outputAudio");
outputAudio.addEventListener("loadedmetadata", () => {
  $("#transportDuration").textContent = formatClock(outputAudio.duration);
});
outputAudio.addEventListener("timeupdate", () => {
  $("#transportCurrent").textContent = formatClock(outputAudio.currentTime);
  $("#transportFill").style.width = `${outputAudio.duration ? outputAudio.currentTime / outputAudio.duration * 100 : 0}%`;
  updatePlaybackFollow(outputAudio.currentTime);
});
outputAudio.addEventListener("seeked", () => updatePlaybackFollow(outputAudio.currentTime, true));
outputAudio.addEventListener("play", () => {
  $(".transport-play").classList.add("playing");
  updatePlaybackFollow(outputAudio.currentTime, false);
});
outputAudio.addEventListener("pause", () => $(".transport-play").classList.remove("playing"));
outputAudio.addEventListener("ended", clearPlaybackFollow);
outputAudio.addEventListener("emptied", clearPlaybackFollow);

refreshProjects()
  .then(() => {
    const requested = new URLSearchParams(location.search).get("project");
    if (requested && projects.some(project => project.id === requested)) return loadProject(requested);
  })
  .catch(error => showToast(error.message, true));
