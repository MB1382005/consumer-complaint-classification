const SAMPLES = [
  "There is a mistake on my credit report and the bureau refused to fix it.",
  "A debt collector keeps harassing me daily about a bill I already paid.",
  "My credit card has an unauthorized charge and an unexpected late fee.",
  "My mortgage payment increased without notice due to an escrow error.",
  "Money was stolen from my checking account through unauthorized ATM withdrawals.",
];

const complaintInput = document.getElementById("complaintInput");
const submitBtn = document.getElementById("submitBtn");
const clearBtn = document.getElementById("clearBtn");
const randomSampleBtn = document.getElementById("randomSampleBtn");
const resultCard = document.getElementById("resultCard");
const errorBanner = document.getElementById("errorBanner");

// Sidebar navigation
document.querySelectorAll(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`view-${btn.dataset.view}`).classList.add("active");
  });
});

// Quick scenario cards
document.querySelectorAll(".quick-card").forEach((card) => {
  card.addEventListener("click", () => {
    complaintInput.value = card.dataset.sample;
    runPrediction();
  });
});

clearBtn.addEventListener("click", () => {
  complaintInput.value = "";
  resultCard.classList.add("hidden");
  errorBanner.classList.add("hidden");
  complaintInput.focus();
});

let sampleIdx = 0;
randomSampleBtn.addEventListener("click", () => {
  complaintInput.value = SAMPLES[sampleIdx % SAMPLES.length];
  sampleIdx += 1;
  complaintInput.focus();
});

submitBtn.addEventListener("click", runPrediction);
complaintInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    runPrediction();
  }
});

async function runPrediction() {
  const text = complaintInput.value.trim();
  if (!text) return;

  submitBtn.disabled = true;
  errorBanner.classList.add("hidden");

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Classification failed.");

    renderResult(data);
  } catch (err) {
    errorBanner.textContent = err.message;
    errorBanner.classList.remove("hidden");
  } finally {
    submitBtn.disabled = false;
  }
}

function renderResult(data) {
  document.getElementById("resDept").textContent = data.department;
  document.getElementById("resDesc").textContent = data.description;
  document.getElementById("resConf").textContent = `${data.confidence.toFixed(2)}%`;
  document.getElementById("resLabel").textContent = data.label;
  document.getElementById("resQueue").textContent = data.queue;
  document.getElementById("resSla").textContent = data.sla;

  const badge = document.getElementById("resRouteBadge");
  if (data.auto_route) {
    badge.textContent = "AUTO-ROUTE";
    badge.classList.remove("review");
  } else {
    badge.textContent = "MANUAL REVIEW";
    badge.classList.add("review");
  }

  const probBars = document.getElementById("probBars");
  probBars.innerHTML = data.breakdown
    .map(
      (item, idx) => `
      <div class="prob-row">
        <div class="prob-row-head">
          <span>${item.title}</span>
          <span style="font-family:'JetBrains Mono',monospace;">${item.score.toFixed(2)}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill ${idx === 0 ? "top" : ""}" style="width: ${Math.max(item.score, 1)}%;"></div>
        </div>
      </div>
    `
    )
    .join("");

  resultCard.classList.remove("hidden");
}

async function initMetadata() {
  try {
    const [statusRes, benchRes] = await Promise.all([
      fetch("/api/status"),
      fetch("/api/benchmark"),
    ]);
    const status = await statusRes.json();
    const bench = await benchRes.json();

    if (status.model_name) {
      document.getElementById("activeModelLabel").textContent =
        `${status.model_name} Transformer · Ready`;
    }

    const deptGrid = document.getElementById("deptGrid");
    deptGrid.innerHTML = Object.entries(status.departments || {})
      .map(
        ([key, d]) => `
        <div class="dept-card">
          <h3>${d.title}</h3>
          <p>${d.desc}</p>
          <div class="dept-tags">
            <span>${key}</span>
            <span>·</span>
            <span>${d.queue}</span>
            <span>·</span>
            <span>SLA ${d.sla}</span>
          </div>
        </div>
      `
      )
      .join("");

    const tbody = document.getElementById("benchmarkBody");
    tbody.innerHTML = (bench.rows || [])
      .map(
        (r) => `
        <tr>
          <td>${r["Model"]}</td>
          <td>${(r["Accuracy"] * 100).toFixed(2)}%</td>
          <td>${(r["Precision (Weighted)"] * 100).toFixed(2)}%</td>
          <td>${(r["Recall (Weighted)"] * 100).toFixed(2)}%</td>
          <td>${(r["F1-Score (Weighted)"] * 100).toFixed(2)}%</td>
          <td>${(r["F1-Score (Macro)"] * 100).toFixed(2)}%</td>
        </tr>
      `
      )
      .join("");
  } catch (e) {
    console.error("Failed to load metadata:", e);
  }
}

initMetadata();
