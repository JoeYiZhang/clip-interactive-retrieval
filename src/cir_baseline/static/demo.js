const state = {
  indexKey: null,
  referenceId: null,
  referenceOffset: 0,
  sessionId: null,
  roundId: 0,
  results: [],
  votes: new Map(),
};

const $ = (id) => document.getElementById(id);
const toast = (message, error = false) => {
  const node = $("toast");
  node.textContent = message;
  node.className = error ? "show error" : "show";
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => { node.className = ""; }, 3600);
};

async function api(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: options.body ? {"Content-Type": "application/json", ...(options.headers || {})} : options.headers,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

function imageCard(item, kind) {
  const card = document.createElement("article");
  card.className = `image-card ${kind}-card`;
  card.dataset.imageId = item.image_id;
  const img = document.createElement("img");
  img.src = item.image_url;
  img.alt = item.image_id;
  img.loading = "lazy";
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = kind === "result" ? `#${item.rank} · ${item.score.toFixed(4)} · ${item.image_id}` : item.image_id;
  card.append(img, meta);
  return card;
}

async function loadReferences(reset = false) {
  if (reset) state.referenceOffset = 0;
  const data = await api(`/v1/demo/indexes/${encodeURIComponent(state.indexKey)}/references?offset=${state.referenceOffset}&limit=24`);
  const grid = $("reference-grid");
  grid.replaceChildren();
  state.referenceId = null;
  $("start-session").disabled = true;
  for (const item of data.items) {
    const card = imageCard(item, "reference");
    card.addEventListener("click", () => {
      grid.querySelectorAll(".selected").forEach(node => node.classList.remove("selected"));
      card.classList.add("selected");
      state.referenceId = item.image_id;
      $("start-session").disabled = false;
    });
    grid.append(card);
  }
  state.referenceOffset = (state.referenceOffset + data.limit) % data.total;
}

function updateVoteCounts() {
  let accepted = 0, rejected = 0;
  for (const value of state.votes.values()) {
    if (value === "accepted") accepted += 1;
    if (value === "rejected") rejected += 1;
  }
  $("accepted-count").textContent = accepted;
  $("rejected-count").textContent = rejected;
}

function setVote(card, imageId, next) {
  const previous = state.votes.get(imageId);
  state.votes.delete(imageId);
  card.classList.remove("accepted", "rejected");
  if (previous !== next) {
    state.votes.set(imageId, next);
    card.classList.add(next);
  }
  updateVoteCounts();
}

function renderResults(payload) {
  state.roundId = payload.session.round_id;
  state.results = payload.results;
  state.votes.clear();
  updateVoteCounts();
  $("round-label").textContent = `会话 ${state.sessionId.slice(-8)} · 第 ${state.roundId + 1} 轮 · ${payload.session.status}`;
  $("result-empty").classList.add("hidden");
  $("feedback-panel").classList.remove("hidden");
  const grid = $("result-grid");
  grid.replaceChildren();
  for (const item of payload.results) {
    const card = imageCard(item, "result");
    const rank = document.createElement("span");
    rank.className = "rank";
    rank.textContent = item.rank;
    const bar = document.createElement("div");
    bar.className = "vote-bar";
    const accept = document.createElement("button");
    accept.className = "accept";
    accept.textContent = "喜欢";
    accept.addEventListener("click", () => setVote(card, item.image_id, "accepted"));
    const reject = document.createElement("button");
    reject.className = "reject";
    reject.textContent = "排除";
    reject.addEventListener("click", () => setVote(card, item.image_id, "rejected"));
    bar.append(accept, reject);
    card.prepend(rank);
    card.append(bar);
    grid.append(card);
  }
}

async function retrieve() {
  const payload = await api(`/v1/demo/sessions/${state.sessionId}/retrieve`, {
    method: "POST",
    body: JSON.stringify({top_k: 12}),
  });
  renderResults(payload);
}

async function startSession() {
  if (!state.referenceId) return;
  $("start-session").disabled = true;
  try {
    const created = await api("/v1/demo/sessions", {
      method: "POST",
      body: JSON.stringify({
        index_key: state.indexKey,
        reference_image_id: state.referenceId,
        modification_text: $("query-text").value,
      }),
    });
    state.sessionId = created.session.session_id;
    $("composition").textContent = `策略 ${created.composition.policy_version}；门控 ${created.composition.gate_group}；图像权重 ${created.composition.image_weight.toFixed(2)}；文本置信度 ${created.composition.text_confidence?.toFixed(5) ?? "不适用"}`;
    await retrieve();
    await refreshMetrics();
  } catch (error) {
    toast(error.message, true);
  } finally {
    $("start-session").disabled = false;
  }
}

async function submitFeedback() {
  const accepted = [], rejected = [];
  for (const [imageId, value] of state.votes.entries()) {
    if (value === "accepted") accepted.push(imageId);
    if (value === "rejected") rejected.push(imageId);
  }
  const feedbackText = $("feedback-text").value.trim();
  if (!accepted.length && !rejected.length && !feedbackText) {
    toast("请至少选择一项结果，或填写文本反馈。", true);
    return;
  }
  $("submit-feedback").disabled = true;
  try {
    const response = await api(`/v1/demo/sessions/${state.sessionId}/feedback`, {
      method: "POST",
      body: JSON.stringify({
        event_id: crypto.randomUUID(),
        expected_round_id: state.roundId,
        shown_image_ids: state.results.map(item => item.image_id),
        accepted_ids: accepted,
        rejected_ids: rejected,
        feedback_text: feedbackText,
      }),
    });
    $("feedback-text").value = "";
    if (response.session.status === "active") {
      await retrieve();
    } else {
      $("round-label").textContent = `会话已完成 · ${response.session.close_reason}`;
      $("feedback-panel").classList.add("hidden");
      toast("会话已达到最大轮次。");
    }
    await refreshMetrics();
  } catch (error) {
    toast(error.message, true);
  } finally {
    $("submit-feedback").disabled = false;
  }
}

async function closeSession() {
  if (!state.sessionId) return;
  try {
    await api(`/v1/demo/sessions/${state.sessionId}/close`, {
      method: "POST",
      body: JSON.stringify({reason: "user_closed_from_demo"}),
    });
    $("feedback-panel").classList.add("hidden");
    $("round-label").textContent = "会话已结束";
    toast("会话已安全关闭。");
  } catch (error) { toast(error.message, true); }
}

async function refreshMetrics() {
  try {
    const data = await api("/v1/demo/metrics");
    $("metrics").textContent = JSON.stringify(data.routes, null, 2);
  } catch (error) { toast(error.message, true); }
}

async function initialize() {
  try {
    const health = await api("/v1/demo/health");
    $("health").textContent = `${health.indexes.length} 个真实索引已加载`;
    $("health").classList.add("ok");
    const select = $("index-select");
    for (const index of health.indexes) {
      const option = document.createElement("option");
      option.value = index.key;
      option.textContent = `${index.category.toUpperCase()} · ${index.item_count.toLocaleString()} 张`;
      select.append(option);
    }
    state.indexKey = select.value;
    select.addEventListener("change", async () => {
      state.indexKey = select.value;
      await loadReferences(true);
    });
    await loadReferences(true);
    await refreshMetrics();
  } catch (error) {
    $("health").textContent = "服务不可用";
    toast(error.message, true);
  }
}

$("more-references").addEventListener("click", () => loadReferences(false).catch(error => toast(error.message, true)));
$("start-session").addEventListener("click", startSession);
$("submit-feedback").addEventListener("click", submitFeedback);
$("close-session").addEventListener("click", closeSession);
$("refresh-metrics").addEventListener("click", refreshMetrics);
initialize();
