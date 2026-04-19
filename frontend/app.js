const form = document.querySelector("#screening-form");
const riskLabel = document.querySelector("#risk-label");
const riskScore = document.querySelector("#risk-score");
const gauge = document.querySelector("#gauge-progress");
const summary = document.querySelector("#summary");
const probabilities = document.querySelector("#probabilities");
const factors = document.querySelector("#factors");
const reportButton = document.querySelector("#report-button");
const sampleHigh = document.querySelector("#sample-high");
const formTitle = document.querySelector("#form-title");
const behaviorTitle = document.querySelector("#behavior-title");

let lastPayload = null;

function payloadFromForm() {
  const data = new FormData(form);
  const payload = {};
  for (const [key, value] of data.entries()) {
    payload[key] = key === "respondent" ? value : Number(value);
  }
  return payload;
}

function colorFor(score) {
  if (score >= 65) return "#b91c1c";
  if (score >= 35) return "#a16207";
  return "#15803d";
}

function renderResult(result) {
  riskLabel.textContent = result.riskLabel;
  riskScore.textContent = `${result.riskScore}%`;
  summary.textContent = result.summary;
  const circumference = 301.59;
  gauge.style.strokeDashoffset = String(circumference - (circumference * result.riskScore) / 100);
  gauge.style.stroke = colorFor(result.riskScore);

  probabilities.innerHTML = Object.entries(result.probabilities)
    .map(([label, probability]) => {
      const percent = Math.round(probability * 100);
      return `
        <div class="prob-row">
          <span>${label}</span>
          <span class="bar"><span style="width:${percent}%"></span></span>
          <strong>${percent}%</strong>
        </div>
      `;
    })
    .join("");

  factors.innerHTML = result.topFactors
    .map((factor) => {
      const sign = factor.contribution > 0 ? "+" : "";
      return `
        <article class="factor">
          <div class="factor-top">
            <span>${factor.label}</span>
            <span>${sign}${factor.contribution}</span>
          </div>
          <p>${factor.text}</p>
        </article>
      `;
    })
    .join("");

  reportButton.disabled = false;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = payloadFromForm();
  lastPayload = payload;
  riskLabel.textContent = "Calculating";
  reportButton.disabled = true;
  try {
    const result = await postJson("/api/predict", payload);
    renderResult(result);
  } catch (error) {
    riskLabel.textContent = "Unable to calculate";
    summary.textContent = error.message;
  }
});

reportButton.addEventListener("click", async () => {
  if (!lastPayload) return;
  reportButton.textContent = "Preparing report";
  reportButton.disabled = true;
  try {
    const result = await postJson("/api/report", lastPayload);
    renderResult(result);
    window.location.href = result.reportUrl;
  } catch (error) {
    summary.textContent = error.message;
  } finally {
    reportButton.textContent = "Download PDF report";
    reportButton.disabled = false;
  }
});

sampleHigh.addEventListener("click", () => {
  const sample = {
    age: 74,
    education_years: 10,
    family_history: 1,
    apoe4: 1,
    mmse: 22,
    moca: 19,
    cdr: 0.5,
    memory_recall: 3,
    orientation: 7,
    daily_function: 5,
    mood_change: 6,
    sleep_quality: 4,
    wandering: 1,
    medication_adherence: 5,
    hypertension: 1,
    diabetes: 0,
    amyloid_beta: 620,
    tau: 540,
  };
  Object.entries(sample).forEach(([key, value]) => {
    const input = form.elements[key];
    if (input) input.value = value;
  });
});

form.addEventListener("change", () => {
  const respondent = new FormData(form).get("respondent");
  if (respondent === "Caregiver") {
    formTitle.textContent = "Tell us about the person you are supporting";
    behaviorTitle.textContent = "Changes you have noticed";
  } else if (respondent === "Clinician") {
    formTitle.textContent = "Enter the patient's screening indicators";
    behaviorTitle.textContent = "Observed behavioral indicators";
  } else {
    formTitle.textContent = "Tell us about your screening indicators";
    behaviorTitle.textContent = "Daily changes";
  }
});
