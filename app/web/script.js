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
