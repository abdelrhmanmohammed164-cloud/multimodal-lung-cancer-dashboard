// app.js -- wires the dashboard UI to the real Flask backend /api/analyze
// which calls the actual tested worker scripts. No result shown here is
// hardcoded; everything comes from the JSON response.

let uploadedFiles = [];
let lastSliceThumbnails = [];

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const uploadStatus = document.getElementById('uploadStatus');
const previewFrame = document.getElementById('previewFrame');
const errorBox = document.getElementById('errorBox');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsRow = document.getElementById('resultsRow');

browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

['dragover', 'dragenter'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.style.background = 'rgba(230,67,122,0.08)'; });
});
['dragleave', 'drop'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.style.background = ''; });
});
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  handleFiles(e.dataTransfer.files);
});

function handleFiles(fileList) {
  uploadedFiles = Array.from(fileList);
  const dcmCount = uploadedFiles.filter(f => f.name.toLowerCase().endsWith('.dcm')).length;
  uploadStatus.textContent = `\u2713 ${uploadedFiles.length} file(s) selected (${dcmCount} .dcm)`;
}

document.getElementById('qaUploadCt').addEventListener('click', () => fileInput.click());
document.getElementById('qaNewPatient').addEventListener('click', resetForm);
document.getElementById('newPatientBtn').addEventListener('click', resetForm);
document.getElementById('qaProjectAnalysis').addEventListener('click', () => window.location.href = '/project-analysis');
document.getElementById('qaHistory').addEventListener('click', () => window.location.href = '/results');
document.getElementById('viewSlicesLink').addEventListener('click', () => {
  if (!lastSliceThumbnails.length) {
    alert('Run an analysis first to see the slices used.');
    return;
  }
  const grid = document.getElementById('slicesGrid');
  grid.innerHTML = lastSliceThumbnails.map((b64, i) =>
    `<div class="slice-cell">
       <img src="data:image/png;base64,${b64}">
       <span>Slice ${i + 1}</span>
     </div>`
  ).join('');
  document.getElementById('slicesModal').style.display = 'flex';
});
document.getElementById('slicesModalClose').addEventListener('click', closeSlicesModal);
document.getElementById('slicesModalBackdrop').addEventListener('click', closeSlicesModal);
function closeSlicesModal() {
  document.getElementById('slicesModal').style.display = 'none';
}

document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', (e) => {
    if (item.dataset.page === 'placeholder') {
      e.preventDefault();
      alert('This section is not implemented in the first version yet.');
    }
  });
});

function resetForm() {
  document.getElementById('patientId').value = '';
  document.getElementById('patientName').value = '';
  document.getElementById('quickAge').value = '';
  document.querySelectorAll('[data-field]').forEach(el => { if (el.tagName === 'INPUT') el.value = 0; });
  uploadedFiles = [];
  lastSliceThumbnails = [];
  lastAnalysisResult = null;
  currentSliceIndex = 0;
  uploadStatus.textContent = '';
  previewFrame.innerHTML = '<span class="preview-placeholder">No scan yet</span>';
  document.getElementById('previewNav').style.display = 'none';
  document.getElementById('downloadPdfBtn').style.display = 'none';
  document.getElementById('confidenceBadge').style.display = 'none';
  document.getElementById('wholelungCard').style.display = 'none';
  resultsRow.style.display = 'none';
  errorBox.style.display = 'none';
}

function collectClinicalValues() {
  const values = {};
  document.querySelectorAll('[data-field]').forEach(el => {
    values[el.dataset.field] = parseFloat(el.value) || 0;
  });
  // sync the quick patient-bar age/gender into the underlying model fields if present
  const age = document.getElementById('quickAge').value;
  const gender = document.getElementById('quickGender').value;
  if (age) values['age'] = parseFloat(age);
  if ('gender' in values || document.querySelector('[data-field="gender"]')) values['gender'] = parseFloat(gender);
  return values;
}

document.getElementById('saveClinicalBtn').addEventListener('click', async () => {
  const patientId = document.getElementById('patientId').value.trim();
  if (!patientId) {
    alert('Please enter a Patient ID before saving.');
    return;
  }
  const btn = document.getElementById('saveClinicalBtn');
  btn.disabled = true;
  const originalText = btn.textContent;
  btn.textContent = 'Saving...';
  try {
    const resp = await fetch('/api/save-clinical-draft', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patient_id: patientId,
        patient_name: document.getElementById('patientName').value,
        clinical_values: collectClinicalValues(),
      }),
    });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      alert(data.error || 'Could not save draft.');
    } else {
      alert(`Clinical data saved for ${patientId}. You can reload it later with the "Load" button.`);
    }
  } catch (err) {
    alert('Network error while saving: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }
});

document.getElementById('loadDraftBtn').addEventListener('click', async () => {
  const patientId = document.getElementById('patientId').value.trim();
  if (!patientId) {
    alert('Enter a Patient ID first, then click Load.');
    return;
  }
  try {
    const resp = await fetch(`/api/load-clinical-draft/${encodeURIComponent(patientId)}`);
    const data = await resp.json();
    if (!resp.ok || data.error) {
      alert(data.error || 'No saved draft found for this Patient ID.');
      return;
    }
    document.getElementById('patientName').value = data.patient_name || '';
    const values = data.clinical_values || {};
    document.querySelectorAll('[data-field]').forEach(el => {
      if (el.dataset.field in values) el.value = values[el.dataset.field];
    });
    alert(`Loaded saved draft from ${data.saved_at}.`);
  } catch (err) {
    alert('Network error while loading: ' + err.message);
  }
});

analyzeBtn.addEventListener('click', async () => {
  errorBox.style.display = 'none';
  if (uploadedFiles.length === 0) {
    showError('Please upload the patient\'s CT images first.');
    return;
  }

  analyzeBtn.textContent = 'Analyzing\u2026';
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append('patient_id', document.getElementById('patientId').value);
  formData.append('patient_name', document.getElementById('patientName').value);
  formData.append('clinical_values', JSON.stringify(collectClinicalValues()));
  const relativePaths = uploadedFiles.map(f => f.webkitRelativePath || f.name);
  formData.append('ct_file_paths', JSON.stringify(relativePaths));
  uploadedFiles.forEach(f => formData.append('ct_files', f));

  try {
    const resp = await fetch('/api/analyze', { method: 'POST', body: formData });
    const data = await resp.json();

    if (!resp.ok || data.error) {
      showError(data.error || 'Analysis failed.');
      return;
    }

    renderResults(data);
  } catch (err) {
    showError('Network/server error: ' + err.message);
  } finally {
    analyzeBtn.textContent = '\uD83D\uDD0D  Analyze';
    analyzeBtn.disabled = false;
  }
});

function showError(msg) {
  errorBox.style.display = 'block';
  errorBox.textContent = msg;
}

function renderResults(data) {
  resultsRow.style.display = 'grid';
  lastSliceThumbnails = data.slice_thumbnails_b64 || [];
  currentSliceIndex = 0;

  if (lastSliceThumbnails.length) {
    renderPreviewSlice();
    document.getElementById('previewNav').style.display = 'flex';
  } else if (data.preview_image_b64) {
    previewFrame.innerHTML = `<img src="data:image/png;base64,${data.preview_image_b64}">`;
  }

  setModelCard('clinical', data.clinical_prob);
  setModelCard('cnn2d', data.old_cnn_prob);
  setModelCard('mil3d', data.mil_prob);

  const wholelungCard = document.getElementById('wholelungCard');
  if (data.wholelung_prob !== null && data.wholelung_prob !== undefined) {
    wholelungCard.style.display = 'block';
    const pct = Math.round(data.wholelung_prob * 100);
    document.getElementById('wholelungScore').textContent = data.wholelung_prob.toFixed(2);
    document.getElementById('wholelungScoreLabel').textContent = pct + '% Risk Score';
    document.getElementById('wholelungBar').style.width = pct + '%';
  } else {
    wholelungCard.style.display = 'none';
  }

  const pct = Math.round(data.final_prob * 100);
  document.getElementById('gaugePct').textContent = pct + '%';
  document.getElementById('riskLabel').textContent = data.risk_label.toUpperCase();

  const confBadge = document.getElementById('confidenceBadge');
  if (data.confidence_label) {
    confBadge.style.display = 'inline-block';
    confBadge.textContent = data.confidence_label;
    confBadge.title = data.confidence_note || '';
    confBadge.className = 'confidence-badge ' +
      (data.confidence_label.includes('Very Low') ? 'conf-vlow'
        : data.confidence_label.includes('Low') ? 'conf-low'
        : data.confidence_label.includes('Moderate') ? 'conf-mod'
        : 'conf-high');
  }

  const TIER_COLORS = ['#3FBF7F', '#E6C93F', '#E6A73F', '#E6437A'];
  const riskColor = TIER_COLORS[data.risk_tier] ?? '#8790B0';
  document.getElementById('riskLabel').style.color = riskColor;

  const gauge = document.getElementById('fusionGauge');
  gauge.style.background = `conic-gradient(${riskColor} ${pct * 3.6}deg, rgba(255,255,255,0.06) ${pct * 3.6}deg)`;

  const TIER_RECOMMENDATIONS = [
    'No indicators of concern were found. Continue routine screening as scheduled.',
    'Some indicators are present but below the model\'s decision threshold. Clinical correlation and routine follow-up are recommended.',
    'This result is above the model\'s decision threshold but not with high confidence. Further evaluation and a follow-up scan are recommended.',
    'Strong indicators of concern were found. Further clinical evaluation and diagnostic follow-up are strongly recommended.',
  ];
  const recText = TIER_RECOMMENDATIONS[data.risk_tier] ?? 'Run an analysis to see a recommendation.';
  document.getElementById('recommendationText').textContent = recText;

  lastAnalysisResult = { ...data, recommendation_text: recText };
  document.getElementById('downloadPdfBtn').style.display = 'inline-block';

  window.scrollTo({ top: resultsRow.offsetTop - 20, behavior: 'smooth' });
}

let currentSliceIndex = 0;
let lastAnalysisResult = null;

function renderPreviewSlice() {
  if (!lastSliceThumbnails.length) return;
  previewFrame.innerHTML = `<img src="data:image/png;base64,${lastSliceThumbnails[currentSliceIndex]}">`;
  document.getElementById('previewSliceLabel').textContent =
    `${currentSliceIndex + 1} / ${lastSliceThumbnails.length}`;
}

document.getElementById('prevSliceBtn').addEventListener('click', () => {
  if (!lastSliceThumbnails.length) return;
  currentSliceIndex = (currentSliceIndex - 1 + lastSliceThumbnails.length) % lastSliceThumbnails.length;
  renderPreviewSlice();
});
document.getElementById('nextSliceBtn').addEventListener('click', () => {
  if (!lastSliceThumbnails.length) return;
  currentSliceIndex = (currentSliceIndex + 1) % lastSliceThumbnails.length;
  renderPreviewSlice();
});

document.getElementById('downloadPdfBtn').addEventListener('click', async () => {
  if (!lastAnalysisResult) return;
  const btn = document.getElementById('downloadPdfBtn');
  btn.disabled = true;
  const originalText = btn.textContent;
  btn.textContent = 'Generating...';
  try {
    const resp = await fetch('/api/report-pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patient_id: document.getElementById('patientId').value || 'patient',
        patient_name: document.getElementById('patientName').value || '—',
        ...lastAnalysisResult,
      }),
    });
    if (!resp.ok) {
      alert('Could not generate PDF report.');
      return;
    }
    const blob = await resp.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `report_${document.getElementById('patientId').value || 'patient'}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    alert('Network error while generating PDF: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }
});

function setModelCard(prefix, prob) {
  const pct = Math.round(prob * 100);
  document.getElementById(prefix + 'Score').textContent = prob.toFixed(2);
  document.getElementById(prefix + 'ScoreLabel').textContent = pct + '% Risk Score';
  document.getElementById(prefix + 'Bar').style.width = pct + '%';
}
