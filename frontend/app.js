document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    // UI Elements
    const btn          = document.getElementById('predict-btn');
    const btnText      = document.getElementById('btn-text');
    const loader       = document.getElementById('btn-loader');
    const resultEmpty  = document.getElementById('result-container');
    const resultData   = document.getElementById('result-data');
    const gaugeFill    = document.getElementById('gauge-fill');
    const probPct      = document.getElementById('prob-percentage');
    const churnStatus  = document.getElementById('churn-status');
    const churnDetail  = document.getElementById('churn-detail');
    const statusCard   = document.getElementById('status-card');

    // Loading state
    btnText.classList.add('hidden');
    loader.classList.remove('hidden');
    btn.disabled = true;

    // Collect form data
    const formData = new FormData(e.target);
    const data = {};
    formData.forEach((value, key) => {
        if (key === 'SeniorCitizen' || key === 'tenure') {
            data[key] = parseInt(value, 10);
        } else if (key === 'MonthlyCharges' || key === 'TotalCharges') {
            data[key] = parseFloat(value);
        } else {
            data[key] = value;
        }
    });

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(err.detail || `API Error ${response.status}`);
        }

        const result = await response.json();

        // Show result panel
        resultEmpty.classList.add('hidden');
        resultData.classList.remove('hidden');

        const probability = Math.round(result.churn_probability * 100);

        // Animate gauge
        const circumference = 125.6;
        const offset = circumference - (probability / 100) * circumference;
        setTimeout(() => { gaugeFill.style.strokeDashoffset = offset; }, 50);

        probPct.innerText = `${probability}%`;

        // Confidence badge
        const confidenceLabel = result.confidence
            ? ` · ${result.confidence.charAt(0).toUpperCase() + result.confidence.slice(1)} confidence`
            : '';

        if (result.churn_prediction === 1) {
            gaugeFill.style.stroke       = 'var(--danger)';
            churnStatus.innerText        = '⚠ High Risk';
            churnStatus.style.color      = 'var(--danger)';
            statusCard.style.borderColor = 'rgba(239, 68, 68, 0.3)';
            churnDetail.innerText        = `Likely to churn${confidenceLabel}`;
        } else {
            gaugeFill.style.stroke       = 'var(--safe)';
            churnStatus.innerText        = '✓ Low Risk';
            churnStatus.style.color      = 'var(--safe)';
            statusCard.style.borderColor = 'rgba(16, 185, 129, 0.3)';
            churnDetail.innerText        = `Likely to stay${confidenceLabel}`;
        }

        // Feature importance (SHAP)
        renderFeatureImportance(result.feature_importance);

    } catch (err) {
        console.error(err);
        alert(`Prediction failed: ${err.message}\n\nMake sure the API is running on port 8000.`);
    } finally {
        btnText.classList.remove('hidden');
        loader.classList.add('hidden');
        btn.disabled = false;
    }
});

function renderFeatureImportance(importance) {
    // Remove old section if present
    const old = document.getElementById('feature-importance-section');
    if (old) old.remove();

    if (!importance || Object.keys(importance).length === 0) return;

    // Sort by absolute value
    const sorted = Object.entries(importance)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 5);

    const maxAbs = Math.max(...sorted.map(([, v]) => Math.abs(v)));

    const section = document.createElement('div');
    section.id = 'feature-importance-section';
    section.style.cssText = 'margin-top:1.5rem; text-align:left;';

    section.innerHTML = `
        <p style="font-size:0.8rem; color:var(--text-muted); margin-bottom:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">
            Top Factors
        </p>
        ${sorted.map(([feat, val]) => {
            const pct   = Math.round((Math.abs(val) / maxAbs) * 100);
            const color = val > 0 ? 'var(--danger)' : 'var(--safe)';
            const label = feat.replace(/_/g, ' ');
            const sign  = val > 0 ? '↑' : '↓';
            return `
            <div style="margin-bottom:0.6rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:0.25rem;">
                    <span style="color:var(--text-main); text-transform:capitalize;">${label}</span>
                    <span style="color:${color}; font-weight:600;">${sign} ${Math.abs(val).toFixed(3)}</span>
                </div>
                <div style="background:rgba(255,255,255,0.05); border-radius:4px; height:6px; overflow:hidden;">
                    <div style="width:${pct}%; height:100%; background:${color}; border-radius:4px; transition:width 0.8s ease;"></div>
                </div>
            </div>`;
        }).join('')}
    `;

    document.getElementById('status-card').after(section);
}
