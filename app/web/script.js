const API_BASE = "http://localhost:8000";

let currentMeetingId = null;
let meetings = {};
let chatHistories = {};
let selectedFile = null;

const fileInput = document.getElementById("fileInput");
const uploadBox = document.getElementById("uploadBox");
const uploadText = document.getElementById("uploadText");
const processBtn = document.getElementById("processBtn");
const uploadStatus = document.getElementById("uploadStatus");
const meetingList = document.getElementById("meetingList");
const emptyState = document.getElementById("emptyState");
const meetingView = document.getElementById("meetingView");
const meetingTitle = document.getElementById("meetingTitle");

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    selectedFile = fileInput.files[0];
    uploadText.textContent = selectedFile.name;
    processBtn.disabled = false;
  }
});

processBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  processBtn.disabled = true;
  uploadStatus.textContent = "Processing... this may take a few minutes.";
  uploadStatus.className = "status-text";
  showProcessingState();

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch(API_BASE + "/meetings/upload", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    if (data.status === "ready") {
      uploadStatus.textContent = "Processed successfully.";
      uploadStatus.className = "status-text success";
      await loadMeeting(data.meeting_id);
      selectFile_reset();
    } else {
      uploadStatus.textContent = "Failed: " + (data.error || "unknown error");
      uploadStatus.className = "status-text error";
    }
  } catch (err) {
    uploadStatus.textContent = "Request failed: " + err.message;
    uploadStatus.className = "status-text error";
  }
  processBtn.disabled = false;
});

function selectFile_reset() {
  selectedFile = null;
  fileInput.value = "";
  uploadText.textContent = "Click to choose an audio file";
  processBtn.disabled = true;
}

async function loadMeeting(meetingId) {
  const res = await fetch(API_BASE + "/meetings/" + meetingId);
  const data = await res.json();
  if (data.status !== "ready") {
    alert("Meeting failed to process: " + (data.error || "unknown error"));
    return;
  }
  meetings[meetingId] = data;
  if (!chatHistories[meetingId]) chatHistories[meetingId] = [];
  currentMeetingId = meetingId;
  renderMeetingList();
  renderMeeting(data);
}

function renderMeetingList() {
  const ids = Object.keys(meetings);
  if (ids.length === 0) {
    meetingList.innerHTML = '<div class="empty-text">No meetings yet</div>';
    return;
  }
  meetingList.innerHTML = "";
  ids.forEach((id) => {
    const div = document.createElement("div");
    div.className = "meeting-item" + (id === currentMeetingId ? " active" : "");
    div.textContent = meetings[id].filename;
    div.addEventListener("click", () => {
      currentMeetingId = id;
      renderMeetingList();
      renderMeeting(meetings[id]);
    });
    meetingList.appendChild(div);
  });
}

function renderMeeting(data) {
  hideProcessingState();
  emptyState.classList.add("hidden");
  meetingView.classList.remove("hidden");
  meetingTitle.textContent = data.filename;

  const s = data.summary;
  document.getElementById("summaryOverview").textContent = s.overview || "No overview available.";

  const kpList = document.getElementById("summaryKeyPoints");
  kpList.innerHTML = s.key_points.length
    ? s.key_points.map((p) => `<li>${escapeHtml(p)}</li>`).join("")
    : '<li style="color:var(--text-faint)">No key points extracted.</li>';

  const decList = document.getElementById("summaryDecisions");
  decList.innerHTML = s.decisions.length
    ? s.decisions.map((d) => `<li>${escapeHtml(d)}</li>`).join("")
    : '<li style="color:var(--text-faint)">No decisions extracted.</li>';

  const aiContainer = document.getElementById("summaryActionItems");
  if (s.action_items.length === 0) {
    aiContainer.innerHTML = '<div style="color:var(--text-faint); font-size:13px;">No action items extracted.</div>';
  } else {
    aiContainer.innerHTML = s.action_items
      .map(
        (item) => `
      <div class="action-item">
        <div>
          <div class="task">${escapeHtml(item.task)}</div>
          ${item.evidence ? `<div class="evidence">Evidence: ${escapeHtml(item.evidence)}</div>` : ""}
        </div>
        <div class="tags">
          <span class="badge">Assigned: ${escapeHtml(item.assigned_to)}</span>
          <span class="badge">Due: ${escapeHtml(item.deadline)}</span>
        </div>
      </div>`
      )
      .join("");
  }

  const transcriptContainer = document.getElementById("transcriptContainer");
  transcriptContainer.innerHTML = data.transcript
    .map((seg) => {
      const ts = formatTimestamp(seg.start);
      const speaker = seg.speaker ? `<strong>${escapeHtml(seg.speaker)}:</strong> ` : "";
      return `<div class="transcript-line"><span class="ts-chip">${ts}</span>${speaker}${escapeHtml(seg.text)}</div>`;
    })
    .join("");

  renderChat();
}

function formatTimestamp(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// Tabs
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "analytics" && currentMeetingId) {
      loadAnalytics(currentMeetingId);
    }
  });
});

// Chat
const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const chatSendBtn = document.getElementById("chatSendBtn");

function renderChat() {
  const history = chatHistories[currentMeetingId] || [];
  chatMessages.innerHTML = "";
  history.forEach((entry) => appendChatMessage(entry.question, entry.answer, entry.evidence));
}

function appendChatMessage(question, answer, evidence) {
  const userDiv = document.createElement("div");
  userDiv.className = "chat-msg user";
  userDiv.textContent = question;
  chatMessages.appendChild(userDiv);

  const assistantDiv = document.createElement("div");
  assistantDiv.className = "chat-msg assistant";
  assistantDiv.textContent = answer;

  if (evidence && evidence.length > 0) {
    const toggle = document.createElement("div");
    toggle.className = "evidence-toggle";
    toggle.textContent = "Show evidence";
    const block = document.createElement("div");
    block.className = "evidence-block";
    block.innerHTML = evidence
      .map((e) => `<div class="transcript-line"><span class="ts-chip">${formatTimestamp(e.start)}</span>${escapeHtml(e.text)}</div>`)
      .join("");
    toggle.addEventListener("click", () => {
      block.classList.toggle("show");
      toggle.textContent = block.classList.contains("show") ? "Hide evidence" : "Show evidence";
    });
    assistantDiv.appendChild(toggle);
    assistantDiv.appendChild(block);
  }

  chatMessages.appendChild(assistantDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendQuestion() {
  const question = chatInput.value.trim();
  if (!question || !currentMeetingId) return;

  chatInput.value = "";
  chatSendBtn.disabled = true;

  const thinkingDiv = document.createElement("div");
  thinkingDiv.className = "chat-msg assistant";
  thinkingDiv.textContent = "Thinking...";
  const userDiv = document.createElement("div");
  userDiv.className = "chat-msg user";
  userDiv.textContent = question;
  chatMessages.appendChild(userDiv);
  chatMessages.appendChild(thinkingDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  try {
    const res = await fetch(API_BASE + "/meetings/" + currentMeetingId + "/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    thinkingDiv.remove();
    userDiv.remove();

    chatHistories[currentMeetingId].push({ question, answer: data.answer, evidence: data.evidence });
    appendChatMessage(question, data.answer, data.evidence);
  } catch (err) {
    thinkingDiv.textContent = "Error: " + err.message;
  }
  chatSendBtn.disabled = false;
}

chatSendBtn.addEventListener("click", sendQuestion);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendQuestion();
});

// Load existing meetings on page load
(async function init() {
  try {
    const res = await fetch(API_BASE + "/meetings");
    const list = await res.json();
    for (const m of list) {
      if (m.status === "ready") {
        await loadMeeting(m.meeting_id);
      }
    }
    currentMeetingId = null;
    if (Object.keys(meetings).length > 0) {
      const firstId = Object.keys(meetings)[0];
      currentMeetingId = firstId;
      renderMeetingList();
      renderMeeting(meetings[firstId]);
    }
  } catch (err) {
    console.error("Could not load existing meetings:", err);
  }
})();

const exportPdfBtn = document.getElementById("exportPdfBtn");
exportPdfBtn.addEventListener("click", () => {
  if (!currentMeetingId) return;
  window.open(API_BASE + "/meetings/" + currentMeetingId + "/export/pdf", "_blank");
});

let charts = { speaker: null, keyword: null, activity: null };

async function loadAnalytics(meetingId) {
  try {
    const res = await fetch(API_BASE + "/meetings/" + meetingId + "/analytics");
    const data = await res.json();
    renderAnalytics(data);
  } catch (err) {
    console.error("Could not load analytics:", err);
  }
}

function renderAnalytics(data) {
  const mins = Math.floor(data.duration_seconds / 60);
  const secs = Math.floor(data.duration_seconds % 60);
  document.getElementById("statDuration").textContent = `${mins}:${String(secs).padStart(2, "0")}`;
  document.getElementById("statWords").textContent = data.word_count.toLocaleString();
  document.getElementById("statWpm").textContent = data.words_per_minute;

  const chartColors = ["#7c6df2", "#ff9dc9", "#6ee7b7", "#f9d372", "#82b8f7", "#f79a82"];

  if (charts.speaker) charts.speaker.destroy();
  const speakerCtx = document.getElementById("speakerChart").getContext("2d");
  charts.speaker = new Chart(speakerCtx, {
    type: "doughnut",
    data: {
      labels: data.speaker_stats.map((s) => s.speaker),
      datasets: [{
        data: data.speaker_stats.map((s) => s.percentage),
        backgroundColor: chartColors,
        borderWidth: 0,
      }],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { color: "#b4b6c2", font: { size: 11 } } } },
    },
  });

  if (charts.keyword) charts.keyword.destroy();
  const keywordCtx = document.getElementById("keywordChart").getContext("2d");
  charts.keyword = new Chart(keywordCtx, {
    type: "bar",
    data: {
      labels: data.top_keywords.map((k) => k.word),
      datasets: [{
        label: "Mentions",
        data: data.top_keywords.map((k) => k.count),
        backgroundColor: "#7c6df2",
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#8b8d9a" }, grid: { color: "rgba(255,255,255,0.06)" } },
        y: { ticks: { color: "#b4b6c2" }, grid: { display: false } },
      },
    },
  });

  if (charts.activity) charts.activity.destroy();
  const activityCtx = document.getElementById("activityChart").getContext("2d");
  charts.activity = new Chart(activityCtx, {
    type: "line",
    data: {
      labels: data.activity_buckets.map((b) => b.label),
      datasets: [{
        label: "Words spoken",
        data: data.activity_buckets.map((b) => b.word_count),
        borderColor: "#7c6df2",
        backgroundColor: "rgba(124,109,242,0.15)",
        fill: true,
        tension: 0.3,
      }],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#8b8d9a" }, grid: { display: false } },
        y: { ticks: { color: "#8b8d9a" }, grid: { color: "rgba(255,255,255,0.06)" } },
      },
    },
  });
}




const processingTips = [
  "Transcribing audio with Whisper...",
  "Identifying who spoke when...",
  "Cleaning up the transcript...",
  "Generating embeddings...",
  "Summarizing key points...",
  "Almost there...",
];

let processingTipInterval = null;

function showProcessingState() {
  emptyState.classList.add("hidden");
  meetingView.classList.add("hidden");

  let existing = document.getElementById("processingState");
  if (!existing) {
    existing = document.createElement("div");
    existing.id = "processingState";
    existing.className = "processing-state";
    existing.innerHTML = `
      <div class="spinner"></div>
      <div>Processing your meeting...</div>
      <div class="processing-tip" id="processingTipText">${processingTips[0]}</div>
    `;
    document.querySelector(".main").appendChild(existing);
  }
  existing.classList.remove("hidden");

  let tipIndex = 0;
  processingTipInterval = setInterval(() => {
    tipIndex = (tipIndex + 1) % processingTips.length;
    const tipEl = document.getElementById("processingTipText");
    if (tipEl) {
      tipEl.style.opacity = 0;
      setTimeout(() => {
        tipEl.textContent = processingTips[tipIndex];
        tipEl.style.opacity = 1;
      }, 300);
    }
  }, 2500);
}

function hideProcessingState() {
  const el = document.getElementById("processingState");
  if (el) el.classList.add("hidden");
  if (processingTipInterval) clearInterval(processingTipInterval);
}


