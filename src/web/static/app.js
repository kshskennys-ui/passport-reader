const state = {
  jobId: null,
  pages: new Map(),
  artifacts: new Map(),
  selectedPage: null,
  eventSource: null,
};

const $ = (id) => document.getElementById(id);
const fileInput = $("fileInput");
const dropzone = $("dropzone");
const startButton = $("startButton");
const imageViewer = $("imageViewer");
const viewerImage = $("viewerImage");
const viewerCaption = $("viewerCaption");
let selectedFile = null;

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
dropzone.addEventListener("dragover", (event) => { event.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  setFile(event.dataTransfer.files[0]);
});
startButton.addEventListener("click", startJob);
$("viewerClose").addEventListener("click", closeViewer);
imageViewer.addEventListener("click", (event) => { if (event.target === imageViewer) closeViewer(); });
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeViewer(); });

function setFile(file) {
  if (!file) return;
  const allowed = /\.(pdf|jpe?g|png|tiff?)$/i.test(file.name);
  if (!allowed) { addLog("不支持的文件类型"); return; }
  selectedFile = file;
  $("fileName").textContent = file.name;
  startButton.disabled = false;
  setStatus("idle", "已选择文件");
}

async function startJob() {
  if (!selectedFile) return;
  startButton.disabled = true;
  resetView();
  const body = new FormData();
  body.append("file", selectedFile);
  body.append("workers", $("workers").value);
  try {
    const response = await fetch("/api/jobs", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "任务创建失败");
    state.jobId = payload.job_id;
    setStatus("running", "任务已创建");
    subscribe(payload.job_id);
  } catch (error) {
    setStatus("error", error.message);
    startButton.disabled = false;
  }
}

function subscribe(jobId) {
  if (state.eventSource) state.eventSource.close();
  state.eventSource = new EventSource(`/api/jobs/${jobId}/events`);
  state.eventSource.onopen = () => { $("connectionText").textContent = "实时连接"; };
  state.eventSource.onerror = () => { $("connectionText").textContent = "连接重试中"; };
  state.eventSource.onmessage = (message) => handleEvent(JSON.parse(message.data));
}

function handleEvent(event) {
  if (event.type === "status") {
    setStatus(event.status, event.message);
    return;
  }
  if (event.type === "log") { addLog(event.message); return; }
  if (event.type === "artifact") { addArtifact(event); return; }
  if (event.type === "phase1_complete") {
    $("dataPageCount").textContent = event.selected;
    addLog(`个人信息页提取完成：${event.selected} / ${event.pages}`);
    return;
  }
  if (event.type === "mrz_page") { updateMrzPage(event); return; }
  if (event.type === "report") { $("reportLink").href = event.url; $("reportLink").classList.remove("hidden"); return; }
  if (event.type === "export") { $("exportLink").href = event.url; $("exportLink").classList.remove("hidden"); addLog("Excel 已生成，可下载"); return; }
  if (event.type === "export_error") { addLog(`Excel 导出失败：${event.message}`); return; }
  if (event.type === "error") { addLog(`错误：${event.message}`); return; }
  if (event.type === "done") {
    startButton.disabled = false;
    if (state.eventSource) state.eventSource.close();
  }
}

function addArtifact(event) {
  const key = `${event.page || "job"}:${event.stage}:${event.path}`;
  state.artifacts.set(key, event);
  if (event.page) {
    const page = state.pages.get(event.page) || { page: event.page, artifacts: [], parse: null };
    if (!page.artifacts.some((item) => item.path === event.path)) page.artifacts.push(event);
    state.pages.set(event.page, page);
    renderPageList();
    if (state.selectedPage === null) selectPage(event.page);
    else if (state.selectedPage === event.page) renderDetail(page);
  }
}

function updateMrzPage(event) {
  const page = state.pages.get(event.page) || { page: event.page, artifacts: [], parse: null };
  page.parse = event.parse;
  page.status = event.status;
  page.mode = event.mode;
  page.elapsed = event.elapsed_ms;
  page.warnings = event.warnings || [];
  page.files = event.files || {};
  state.pages.set(event.page, page);
  $("progressText").textContent = `${event.index} / ${event.total}`;
  $("progressBar").style.width = `${Math.round(event.index / event.total * 100)}%`;
  renderPageList();
  if (state.selectedPage === null || state.selectedPage === event.page) selectPage(event.page);
  $("validCount").textContent = [...state.pages.values()].filter((item) => item.parse?.status === "valid").length;
  $("warningCount").textContent = [...state.pages.values()].filter((item) => item.parse && item.parse.status !== "valid").length;
}

function selectPage(pageNumber) {
  state.selectedPage = pageNumber;
  const page = state.pages.get(pageNumber);
  if (page) renderDetail(page);
  renderPageList();
}

function renderPageList() {
  const pages = [...state.pages.values()].sort((a, b) => a.page - b.page);
  $("pageCountLabel").textContent = `${pages.length} 页`;
  if (!pages.length) { $("pageList").innerHTML = '<div class="empty-state">等待页面结果</div>'; return; }
  $("pageList").innerHTML = pages.map((page) => {
    const parse = page.parse?.status || "处理中";
    const cls = parse === "valid" ? "ok" : parse === "处理中" ? "" : "warning";
    return `<div class="page-item ${page.page === state.selectedPage ? "selected" : ""}" data-page="${page.page}">
      <span class="page-number">${String(page.page).padStart(3, "0")}</span>
      <span><strong>第 ${page.page} 页</strong><small>${page.mode || "正在提取图像"}</small></span>
      <span class="page-state ${cls}">${parse === "valid" ? "通过" : parse}</span>
    </div>`;
  }).join("");
  document.querySelectorAll(".page-item").forEach((item) => item.addEventListener("click", () => selectPage(Number(item.dataset.page))));
}

function renderDetail(page) {
  $("detailTitle").textContent = `第 ${page.page} 页`;
  $("detailStage").textContent = page.mode || "图像处理中";
  const artifacts = page.artifacts || [];
  const byPath = new Map(artifacts.map((item) => [item.path, item]));
  const preferred = [...byPath.values()].slice(-8);
  const urls = [...preferred].map((item) => `<figure class="artifact"><img src="${item.url}" loading="lazy" alt="${item.stage}"><figcaption>${item.stage}</figcaption></figure>`).join("");
  $("imageStrip").innerHTML = urls || '<div class="empty-state">该页图像尚未生成</div>';
  document.querySelectorAll(".artifact").forEach((item) => item.addEventListener("click", () => {
    const image = item.querySelector("img");
    openViewer(image.src, item.querySelector("figcaption")?.textContent || "图片预览");
  }));
  renderFields(page.parse);
}

function openViewer(url, caption) {
  viewerImage.src = url;
  viewerCaption.textContent = caption;
  imageViewer.classList.remove("hidden");
}
function closeViewer() {
  imageViewer.classList.add("hidden");
  viewerImage.removeAttribute("src");
}

function renderFields(parsed) {
  const badge = $("parseBadge");
  if (!parsed) { badge.className = "badge idle"; badge.textContent = "未完成"; $("fields").innerHTML = '<div class="empty-state">MRZ 解析完成后显示信息</div>'; return; }
  const status = parsed.status || "unknown";
  badge.className = `badge ${status === "valid" ? "ok" : "warning"}`;
  badge.textContent = status === "valid" ? "校验通过" : status === "partial" ? "部分通过" : status;
  const fields = parsed.fields || {};
  const items = [["姓名", [fields.surname, fields.given_names].filter(Boolean).join(" ")], ["护照号码", fields.passport_number], ["国籍", fields.nationality], ["出生日期", formatMrzDate(fields.date_of_birth)], ["性别", formatSex(fields.sex)], ["有效期", formatMrzDate(fields.date_of_expiry)], ["签发国", fields.issuing_state], ["证件类型", fields.document_code]];
  $("fields").innerHTML = items.map(([label, value]) => `<div class="field-cell"><span>${label}</span><strong>${escapeHtml(value || "")}</strong></div>`).join("");
  const checks = parsed.validation?.check_digits || [];
  const passed = checks.filter((item) => item.valid).length;
  const checksBox = $("checks");
  checksBox.className = `checks ${status === "valid" ? "" : "warning"}`;
  checksBox.textContent = `校验项：${passed} / ${checks.length} 通过${parsed.reconstruction?.recovery ? ` · ${parsed.reconstruction.recovery}` : ""}`;
}

function formatMrzDate(value) {
  if (!/^\d{6}$/.test(value || "")) return value || "";
  const year = Number(value.slice(0, 2));
  const fullYear = year >= 50 ? 1900 + year : 2000 + year;
  return `${fullYear}/${value.slice(2, 4)}/${value.slice(4, 6)}`;
}
function formatSex(value) { return value === "male" ? "男" : value === "female" ? "女" : value || ""; }
function escapeHtml(value) { return String(value).replace(/[&<>\"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[char])); }
function addLog(message) { const box = $("log"); box.textContent += `${message}\n`; box.scrollTop = box.scrollHeight; }
function setStatus(status, message) { const badge = $("statusBadge"); const style = status === "done" ? "ok" : status === "queued" ? "idle" : status; badge.className = `badge ${style}`; badge.textContent = message; $("statusText").textContent = message; }
function resetView() { state.pages.clear(); state.artifacts.clear(); state.selectedPage = null; $("pageList").innerHTML = '<div class="empty-state">等待页面结果</div>'; $("imageStrip").innerHTML = '<div class="empty-state">等待中间图像</div>'; $("fields").innerHTML = '<div class="empty-state">MRZ 解析完成后显示信息</div>'; $("checks").className = "checks"; $("checks").textContent = ""; $("log").textContent = ""; $("reportLink").classList.add("hidden"); $("exportLink").classList.add("hidden"); $("progressBar").style.width = "0"; $("progressText").textContent = "0 / 0"; $("validCount").textContent = "0"; $("warningCount").textContent = "0"; }
