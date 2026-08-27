let current = null;
let projects = [];
let timer = null;
let projectFilter = "all";
let selectedRoleId = null;
let lastJobs = {};
let toastTimer = null;
const previewJobKeys = new Set();

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
function setHidden(target, hidden) {
  const node = typeof target === "string" ? $(target) : target;
  if (node) node.hidden = Boolean(hidden);
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
  if (project.pipeline?.status === "done") return "已完成，可试听导出";
  if (project.pipeline?.status === "failed") return "处理失败，可重试";
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
    return `<button class="project-item ${type} ${current?.id === project.id ? "active" : ""}" type="button" onclick="loadProject('${project.id}')">
      <i class="project-notch"></i><span><b>${esc(project.name)}</b><small>${esc(projectMessage(project))}</small></span>
    </button>`;
  }).join("") : `<p class="project-empty">这里还没有${projectFilter === "all" ? "项目" : projectFilter === "voice" ? "脚本配音" : "视频配音"}。</p>`;
}

function setProjectFilter(value) {
  projectFilter = value;
  $$(".filter-tabs button").forEach(button => button.classList.toggle("active", button.dataset.filter === value));
  refreshProjects();
}

async function loadProject(id) {
  current = await api(`/api/projects/${id}`);
  selectedRoleId = current.roles?.some(role => role.id === selectedRoleId) ? selectedRoleId : current.roles?.[0]?.id || null;
  const url = new URL(location.href);
  url.searchParams.set("project", id);
  history.replaceState(null, "", url);
  render();
  await refreshProjects();
  poll();
}

function openCreateDialog() {
  $("#createDialog").showModal();
  setTimeout(() => $("#newProjectName").select(), 30);
}

function showCreateDocument(file) {
  $("#createDocumentName").textContent = file?.name || "导入 TXT / Word / PDF";
  if (file && ($("#newProjectName").value === "未命名配音" || !$("#newProjectName").value.trim())) {
    $("#newProjectName").value = file.name.replace(/\.[^.]+$/, "");
  }
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
  $("#createDocumentState").classList.remove("error");
  $("#createDocumentState").textContent = "有文稿就拖进来拆成台词；没有文稿就直接创建空白对白台。";
  $("#newProjectName").value = "未命名配音";
  $("#createProjectButton").disabled = false;
}

function openVideoDialog() {
  $("#videoDialog").showModal();
}

function openDocumentDialog(intoCurrent = false) {
  intoCurrent = Boolean(intoCurrent && current && current.project_type === "voice");
  $("#documentDialog").dataset.target = intoCurrent ? "current" : "new";
  $("#documentDialogTitle").textContent = intoCurrent ? "导入到当前项目" : "导入文档脚本";
  $("#documentDialogHint").textContent = intoCurrent
    ? "文稿会拆成台词，并按上下文给每句分配情绪与音色"
    : "按段落拆成台词，再按上下文给每个角色和每句话分配音色与情绪";
  setHidden("#documentNameField", intoCurrent);
  setHidden("#documentLangField", intoCurrent);
  $("#documentUploadState").classList.remove("error");
  $("#documentUploadState").textContent = intoCurrent
    ? (isPlaceholderScript(current)
      ? "当前还是空白台词，导入后会替换占位句，并自动分配情绪与音色。"
      : "会保留现有台词，把文档内容追加到后面，只给新台词分配情绪。同名角色会复用。")
    : "保留原文顺序。默认归入旁白；明确的“角色：台词”会新建角色。导入后自动分配情绪与音色，不会立刻生成音频。";
  $("#documentUploadButton").innerHTML = `${icon("upload")}${intoCurrent ? "导入到当前剪辑台" : "导入到对白剪辑台"}`;
  $("#documentDialog").showModal();
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
  const formData = new FormData();
  formData.append("document", file);
  if (!intoCurrent) {
    formData.append("name", $("#documentProjectName").value.trim());
    formData.append("target_lang", $("#documentLang").value);
  }
  const button = $("#documentUploadButton");
  const note = $("#documentUploadState");
  button.disabled = true;
  note.classList.remove("error");
  note.textContent = "正在读取文档、拆成台词并分配情绪与音色，请保持窗口开启……";
  try {
    const url = intoCurrent ? `/api/projects/${current.id}/script-upload` : "/api/script-upload";
    const project = await api(url, {method: "POST", body: formData});
    $("#documentDialog").close();
    button.disabled = false;
    $("#documentFile").value = "";
    $("#documentFileName").textContent = "选择或拖入文稿";
    $("#documentProjectName").value = "";
    await loadProject(project.id);
    const imported = project.script_import || {};
    showToast(`已导入 ${imported.cue_count || project.cues?.length || 0} 条台词，正在按上下文分配情绪与音色`);
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
  const button = $("#createProjectButton");
  const note = $("#createDocumentState");
  button.disabled = true;
  note.classList.remove("error");
  try {
    let project;
    if (file) {
      note.textContent = "正在读取文档、拆成台词并分配情绪与音色，请保持窗口开启……";
      const formData = new FormData();
      formData.append("document", file);
      formData.append("name", name);
      formData.append("target_lang", language);
      project = await api("/api/script-upload", {method: "POST", body: formData});
    } else {
      project = await api("/api/projects", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({project_type: "voice", name, target_lang: language}),
      });
    }
    $("#createDialog").close();
    resetCreateDialog();
    await loadProject(project.id);
    const imported = project.script_import || {};
    showToast(file ? `已导入 ${imported.cue_count || project.cues?.length || 0} 条台词，正在按上下文分配情绪与音色` : "脚本配音项目已创建");
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
  renderSettings();
}

function languageName(value) {
  return {zh: "中文", en: "英语", ja: "日语", ko: "韩语"}[value] || value || "中文";
}

function renderStatus() {
  const pipeline = current.pipeline || {};
  const entries = Object.entries(lastJobs || {}).filter(([key]) => !key.startsWith("preview-"));
  const jobs = entries.map(([, job]) => job);
  const runningJob = jobs.find(job => job.status === "running");
  const failedJob = entries.filter(([key]) => key !== "direction").map(([, job]) => job).find(job => job.status === "failed");
  let status = failedJob ? "failed" : runningJob ? "running" : (pipeline.status || "pending");
  if (current.project_type === "voice" && current.output_audio && !failedJob && !runningJob) status = "done";
  const percent = status === "done" ? 100 : Number(pipeline.percent || (runningJob ? 64 : 0));
  const message = failedJob?.message || runningJob?.message || pipeline.message || current.last_message || "等待编辑";
  const strip = $("#statusStrip");
  if (strip) strip.className = `status-strip ${status}`;
  const statusText = $("#statusText");
  if (statusText) statusText.textContent = message;
  const progress = $("#statusProgress");
  if (progress) progress.style.width = `${percent}%`;
  const percentLabel = $("#statusPercent");
  if (percentLabel) percentLabel.textContent = `${percent}%`;
  setHidden("#retryButton", status !== "failed" || current.project_type === "voice");
  const directing = lastJobs?.direction?.status === "running";
  const generating = entries.some(([key, job]) => key !== "direction" && job.status === "running") || (pipeline.status === "running" && !directing);
  const generateButton = $("#generateButton");
  if (generateButton) {
    generateButton.disabled = Boolean(generating);
    const actionName = current.project_type === "video" ? "生成成片" : "生成整轨";
    generateButton.innerHTML = generating ? `${icon("wave")}生成中…` : `${icon("wave")}${actionName}`;
  }
  const hasOutput = current.project_type === "voice" ? current.output_audio : current.output;
  setHidden("#downloadButton", !hasOutput);
}

function renderTransport() {
  const audio = $("#outputAudio");
  const hasAudio = current.project_type === "voice" ? Boolean(current.output_audio) : Boolean(current.output);
  const newSource = hasAudio ? `/api/projects/${current.id}/preview?v=${encodeURIComponent(current.updated_at || "")}` : "";
  if (newSource && !audio.src.includes(`/api/projects/${current.id}/preview`)) {
    audio.src = newSource;
    audio.load();
  }
  if (!hasAudio) {
    audio.pause();
    audio.removeAttribute("src");
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

function roleOptions(selected) {
  return (current.roles || []).map(role => `<option value="${esc(role.id)}" ${role.id === selected ? "selected" : ""}>${esc(role.name)}</option>`).join("");
}

function renderCues() {
  const cues = current.cues || [];
  $("#scriptSummary").textContent = `${cues.filter(cue => cue.enabled !== false).length} 条启用台词 · ${current.roles?.length || 0} 个角色`;
  setHidden("#emptyCueAction", cues.length > 0);
  $("#cues").innerHTML = cues.map((cue, index) => {
    const role = current.roles?.find(item => item.id === cue.role) || current.roles?.[0] || {};
    const fixed = (current.settings?.timeline_mode || "fixed") === "fixed";
    return `<article class="cue-strip ${cue.enabled === false ? "disabled" : ""}">
      <div class="cue-number">
        <b>${String(index + 1).padStart(2, "0")}</b>
        <div class="cue-reorder">
          <button type="button" onclick="moveCue(${index},-1)" aria-label="上移台词">${icon("up")}</button>
          <button type="button" onclick="moveCue(${index},1)" aria-label="下移台词">${icon("down")}</button>
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
          ${cue.source_text ? `<span class="cue-source" title="${esc(cue.source_text)}">原文：${esc(cue.source_text)}</span>` : ""}
        </div>
      </div>
      <div class="cue-control">
        <button class="button quiet small preview-cue" type="button" onclick="previewCue(${index})">${icon("play")}试听此句</button>
        <div class="timing">
          ${fixed ? `<input type="number" step=".001" value="${cue.start}" aria-label="开始时间" onchange="setCue(${index},'start',+this.value)"><input type="number" step=".001" value="${cue.end}" aria-label="结束时间" onchange="setCue(${index},'end',+this.value)">` : `<input type="number" min="0" step=".05" value="${Number(cue.gap_after ?? .25).toFixed(2)}" aria-label="句后间隔" onchange="setCue(${index},'gap_after',+this.value)" title="句后间隔（秒）">`}
          <button class="icon-button danger" type="button" onclick="removeCue(${index})" aria-label="删除台词">${icon("trash")}</button>
        </div>
      </div>
    </article>`;
  }).join("");
}

function setCue(index, key, value) {
  current.cues[index][key] = value;
  if (key === "role") renderCues();
}

function addCue() {
  if (!current.roles?.length) addRole();
  const last = current.cues?.at(-1);
  const start = last ? Number(last.end || 0) + Number(last.gap_after || 0) : 0;
  current.cues.push({
    id: current.cues.length + 1,
    start,
    end: start + 3,
    source_text: "",
    text: "",
    role: current.roles[0]?.id || "",
    emotion: "自然",
    speed: 1,
    gap_after: Number(current.settings?.default_gap ?? .25),
    enabled: true,
  });
  renderCues();
  setTimeout(() => $$(".cue-strip textarea").at(-1)?.focus(), 20);
}

function removeCue(index) {
  current.cues.splice(index, 1);
  renderCues();
}

function moveCue(index, direction) {
  const target = index + direction;
  if (target < 0 || target >= current.cues.length) return;
  [current.cues[index], current.cues[target]] = [current.cues[target], current.cues[index]];
  renderCues();
}

function renderRoles() {
  const roles = current.roles || [];
  if (!roles.some(role => role.id === selectedRoleId)) selectedRoleId = roles[0]?.id || null;
  $("#roleCount").textContent = roles.length;
  $("#roleTabs").innerHTML = roles.map(role => `<button class="role-tab ${role.mode === "clone" ? "clone" : ""} ${role.id === selectedRoleId ? "active" : ""}" type="button" onclick="selectRole('${role.id}')"><i></i><span>${esc(role.name)}</span></button>`).join("");
  const role = activeRole();
  if (!role) {
    $("#roleInspector").innerHTML = `<div class="inspector-empty">添加一个角色，为其设计新声音或绑定获得授权的参考音频。</div>`;
    return;
  }
  const roleIndex = roles.findIndex(item => item.id === role.id);
  const clone = role.mode === "clone";
  const previewKey = `${current.id}:${role.id}`;
  const previewRunning = previewJobKeys.has(previewKey) || lastJobs[`preview-${role.id}`]?.status === "running";
  $("#roleInspector").innerHTML = `
    <label class="field"><span>角色名称</span><input value="${esc(role.name)}" onchange="setRole(${roleIndex},'name',this.value,true)"></label>
    <div class="mode-switch" role="group" aria-label="声线来源">
      <button class="${clone ? "" : "active"}" type="button" onclick="setRoleMode(${roleIndex},'design')">${icon("spark")}场景造声</button>
      <button class="clone ${clone ? "active" : ""}" type="button" onclick="setRoleMode(${roleIndex},'clone')">${icon("clone")}授权克隆</button>
    </div>
    ${clone ? renderCloneFields(role, roleIndex) : renderDesignFields(role, roleIndex)}
    <div class="preview-block">
      <label class="field"><span>声线试听台词</span><textarea id="rolePreviewText" placeholder="输入一句用于确认声线的台词">${esc(previewTextFor(role.id))}</textarea></label>
      <div class="preview-actions"><button id="rolePreviewButton" class="button primary" type="button" onclick="previewRole()" ${previewRunning ? "disabled" : ""}>${previewRunning ? `${icon("wave")}生成试听中…` : `${icon("play")}生成试听`}</button></div>
      <audio id="rolePreviewAudio" class="role-preview-audio" controls hidden></audio>
    </div>
    ${roles.length > 1 ? `<button class="button quiet danger role-danger" type="button" onclick="removeRole(${roleIndex})">${icon("trash")}删除这个角色</button>` : ""}
  `;
}

function renderDesignFields(role, index) {
  return `
    <div class="field"><span>快速起稿</span><div class="preset-grid">${Object.entries(localVoiceOptions).map(([key, label]) => `<button class="${role.voice === key ? "active" : ""}" type="button" onclick="applyVoicePreset(${index},'${key}')">${esc(label)}</button>`).join("")}</div></div>
    <label class="field"><span>人物与场景声音描述</span><textarea onchange="setRole(${index},'description',this.value)">${esc(role.description || "")}</textarea><small class="field-note">可写年龄、音高、质感、性格、场景、口音和表演习惯。整轨中的每句都会沿用这份角色描述，并叠加该句的表演情绪实际生成。</small></label>
  `;
}

function renderCloneFields(role, index) {
  const hasReference = Boolean(role.ref_audio);
  return `
    <label class="consent-box"><input type="checkbox" ${role.clone_consent ? "checked" : ""} onchange="setRole(${index},'clone_consent',this.checked)"><span>我确认已获得声音本人授权或拥有合法使用权，并对生成内容负责。</span></label>
    <div class="field"><span>参考音频</span>
      ${hasReference ? `<div class="reference-ready"><span><b>${esc(role.ref_name || "已上传参考音频")}</b><small>${role.ref_text ? "高质量上下文克隆" : "快速声纹克隆"}</small></span><button class="icon-button danger" type="button" onclick="removeReference('${role.id}')" aria-label="移除参考音频">${icon("trash")}</button></div>` : `<label class="reference-drop"><input type="file" accept="audio/*" onchange="uploadReference('${role.id}',this.files[0])">${icon("upload")}<b>上传 5–15 秒清晰人声</b><span>WAV、MP3、M4A、FLAC 等</span></label>`}
    </div>
    <label class="field"><span>参考音频准确文本（可选）</span><textarea placeholder="逐字填写音频中实际说出的内容" onchange="setRole(${index},'ref_text',this.value)">${esc(role.ref_text || "")}</textarea><small class="field-note">填写准确文本会使用高质量上下文克隆；留空则使用仅声纹的快速克隆，音色相似度可能降低。</small></label>
  `;
}

function previewTextFor(roleId) {
  return current.cues?.find(cue => cue.role === roleId && cue.text)?.text || "这是这位角色的声音测试，请确认音色和表演是否合适。";
}

function selectRole(roleId) {
  selectedRoleId = roleId;
  renderRoles();
}

function setRole(index, key, value, rerender = false) {
  current.roles[index][key] = value;
  if (key === "description") current.roles[index].style = value;
  if (rerender) {
    renderRoles();
    renderCues();
  }
}

function setRoleMode(index, mode) {
  current.roles[index].mode = mode;
  renderRoles();
  renderCues();
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
    ref_audio: "",
    ref_text: "",
    clone_consent: false,
  });
  selectedRoleId = id;
  renderRoles();
  renderCues();
}

function removeRole(index) {
  const role = current.roles[index];
  const replacement = current.roles.find((_, i) => i !== index)?.id;
  current.cues.forEach(cue => { if (cue.role === role.id) cue.role = replacement; });
  current.roles.splice(index, 1);
  selectedRoleId = replacement;
  renderRoles();
  renderCues();
}

async function uploadReference(roleId, file) {
  if (!file) return;
  const role = current.roles.find(item => item.id === roleId);
  if (!role?.clone_consent) {
    showToast("请先确认已获得声音授权", true);
    renderRoles();
    return;
  }
  const formData = new FormData();
  formData.append("audio", file);
  formData.append("consent", "true");
  formData.append("ref_text", role.ref_text || "");
  try {
    showToast("正在处理参考音频……");
    current = await api(`/api/projects/${current.id}/roles/${roleId}/reference`, {method: "POST", body: formData});
    selectedRoleId = roleId;
    render();
    showToast("参考音频已绑定");
  } catch (error) {
    showToast(error.message, true);
  }
}

async function removeReference(roleId) {
  try {
    current = await api(`/api/projects/${current.id}/roles/${roleId}/reference`, {method: "DELETE"});
    selectedRoleId = roleId;
    render();
    showToast("参考音频已移除，角色已切回场景造声");
  } catch (error) {
    showToast(error.message, true);
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

async function generateProject() {
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
  renderRoles();
  $("#rolePreviewText").value = cue.text;
  await previewRole(cue.emotion || "自然");
}

async function previewRole(emotion = "自然") {
  const role = activeRole();
  const text = $("#rolePreviewText")?.value.trim();
  if (!role || !text) return showToast("请输入试听台词", true);
  const projectId = current.id;
  const previewKey = `${projectId}:${role.id}`;
  if (previewJobKeys.has(previewKey)) return;
  previewJobKeys.add(previewKey);
  renderRoles();
  try {
    await saveProject(true);
    await api(`/api/projects/${projectId}/preview-voice`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({role_id: role.id, text, emotion}),
    });
    pollPreview(projectId, role.id);
  } catch (error) {
    if (error.status === 409) {
      pollPreview(projectId, role.id);
      return;
    }
    previewJobKeys.delete(previewKey);
    showToast(error.message, true);
    renderRoles();
  }
}

async function pollPreview(projectId, roleId) {
  const previewKey = `${projectId}:${roleId}`;
  try {
    const jobs = await api(`/api/projects/${projectId}/jobs`);
    const job = jobs[`preview-${roleId}`];
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
          const file = job.result ? `&file=${encodeURIComponent(job.result)}` : "";
          audio.src = `/api/projects/${projectId}/preview-voice/${roleId}?v=${Date.now()}${file}`;
          setHidden(audio, false);
          audio.play().catch(() => {});
        }
        const button = $("#rolePreviewButton");
        if (button) {
          button.disabled = false;
          button.innerHTML = `${icon("play")}重新生成试听`;
        }
      }
      return;
    }
    setTimeout(() => pollPreview(projectId, roleId), 1200);
  } catch (error) {
    setTimeout(() => pollPreview(projectId, roleId), 1600);
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
    const running = Object.values(jobs).some(job => job.status === "running") || current.pipeline?.status === "running";
    if (running) timer = setTimeout(poll, 1800);
  } catch (error) {
    timer = setTimeout(poll, 2800);
  }
}

function downloadOutput() {
  if (!current) return;
  location.href = `/api/projects/${current.id}/download`;
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

const outputAudio = $("#outputAudio");
outputAudio.addEventListener("loadedmetadata", () => {
  $("#transportDuration").textContent = formatClock(outputAudio.duration);
});
outputAudio.addEventListener("timeupdate", () => {
  $("#transportCurrent").textContent = formatClock(outputAudio.currentTime);
  $("#transportFill").style.width = `${outputAudio.duration ? outputAudio.currentTime / outputAudio.duration * 100 : 0}%`;
});
outputAudio.addEventListener("play", () => $(".transport-play").classList.add("playing"));
outputAudio.addEventListener("pause", () => $(".transport-play").classList.remove("playing"));

refreshProjects()
  .then(() => {
    const requested = new URLSearchParams(location.search).get("project");
    if (requested && projects.some(project => project.id === requested)) return loadProject(requested);
  })
  .catch(error => showToast(error.message, true));
