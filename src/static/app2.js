const API_URL = '/predict';

// Show risk bands relative to the decision threshold returned by the API.
function riskBand(p, t) {
    if (p < t) return { tone: 'low', label: 'Low risk' };
    if (p < (t + 1) / 2) return { tone: 'mid', label: 'Moderate risk' };
    return { tone: 'high', label: 'High risk' };
}

const YN = ['No', 'Yes'];
const NET = ['No', 'Yes', 'No internet service'];
const NET_FIELDS = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'];

const SECTIONS = [
    { title: 'Customer', fields: [
        { name: 'gender', label: 'Gender', type: 'seg', options: ['Female', 'Male'] },
        { name: 'SeniorCitizen', label: 'Senior citizen', type: 'seg', options: [['0', 'No'], ['1', 'Yes']] },
        { name: 'Partner', label: 'Partner', type: 'seg', options: YN },
        { name: 'Dependents', label: 'Dependents', type: 'seg', options: YN },
        { name: 'tenure', label: 'Tenure', type: 'number', value: 1, attrs: 'min="0" max="100" step="1" inputmode="numeric"', suffix: 'months' },
    ]},
    { title: 'Phone and internet', fields: [
        { name: 'PhoneService', label: 'Phone service', type: 'seg', options: YN },
        { name: 'MultipleLines', label: 'Multiple lines', type: 'select', options: ['No phone service', 'No', 'Yes'] },
        { name: 'InternetService', label: 'Internet service', type: 'select', options: ['DSL', 'Fiber optic', 'No'] },
        { name: 'OnlineSecurity', label: 'Online security', type: 'select', options: NET },
        { name: 'OnlineBackup', label: 'Online backup', type: 'select', options: NET },
        { name: 'DeviceProtection', label: 'Device protection', type: 'select', options: NET },
        { name: 'TechSupport', label: 'Tech support', type: 'select', options: NET },
        { name: 'StreamingTV', label: 'Streaming TV', type: 'select', options: NET },
        { name: 'StreamingMovies', label: 'Streaming movies', type: 'select', options: NET },
    ]},
    { title: 'Contract and billing', fields: [
        { name: 'Contract', label: 'Contract', type: 'select', options: ['Month-to-month', 'One year', 'Two year'] },
        { name: 'PaperlessBilling', label: 'Paperless billing', type: 'seg', options: YN },
        { name: 'PaymentMethod', label: 'Payment method', type: 'select', options: ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'] },
        { name: 'MonthlyCharges', label: 'Monthly charges', type: 'number', value: '29.85', attrs: 'min="0" step="0.01" inputmode="decimal"', prefix: '$' },
        { name: 'TotalCharges', label: 'Total charges', type: 'text', value: '29.85', attrs: 'inputmode="decimal"', prefix: '$',
          hint: 'Estimated from tenure and monthly charges until you edit it.' },
    ]},
];

const $ = (id) => document.getElementById(id);
const opt = (o) => (Array.isArray(o) ? o : [o, o]);

function fieldHTML(f) {
    if (f.type === 'seg') {
        const radios = f.options.map((o, i) => {
            const [v, l] = opt(o);
            return `<label><input type="radio" name="${f.name}" value="${v}"${i === 0 ? ' checked' : ''}><span>${l}</span></label>`;
        }).join('');
        return `<fieldset class="field"><legend>${f.label}</legend><div class="seg">${radios}</div></fieldset>`;
    }
    if (f.type === 'select') {
        const opts = f.options.map((o, i) => `<option value="${opt(o)[0]}"${i === 0 ? ' selected' : ''}>${opt(o)[1]}</option>`).join('');
        return `<div class="field"><label class="lbl" for="${f.name}">${f.label}</label><select id="${f.name}" name="${f.name}">${opts}</select></div>`;
    }
    return `<div class="field"><label class="lbl" for="${f.name}">${f.label}</label>
        <div class="input">${f.prefix ? `<span>${f.prefix}</span>` : ''}<input id="${f.name}" name="${f.name}" type="${f.type}" value="${f.value}" ${f.attrs || ''} aria-describedby="err-${f.name}">${f.suffix ? `<span>${f.suffix}</span>` : ''}</div>
        <p class="err" id="err-${f.name}"></p>${f.hint ? `<p class="hint">${f.hint}</p>` : ''}</div>`;
}

$('groups').innerHTML = SECTIONS.map((s) =>
    `<section class="group"><h2>${s.title}</h2><div class="grid">${s.fields.map(fieldHTML).join('')}</div></section>`
).join('');

const form = $('churnForm');
const panel = $('panel');

// Keep service selections valid when the parent service changes.
function lock(select, value, locked) {
    if (locked) select.value = value;
    else if (select.value === value) select.value = 'No';
    select.disabled = locked && select.dataset.lockable !== 'false';
    [...select.options].forEach((o) => { o.disabled = !locked && o.value === value; });
}
function syncDependencies() {
    lock(form.elements.MultipleLines, 'No phone service', form.elements.PhoneService.value === 'No');
    const noNet = form.elements.InternetService.value === 'No';
    NET_FIELDS.forEach((n) => lock(form.elements[n], 'No internet service', noNet));
}

// Update total charges until the user edits that field.
let totalTouched = false;
function syncTotal() {
    if (totalTouched) return;
    const t = Number(form.elements.tenure.value), m = Number(form.elements.MonthlyCharges.value);
    if (Number.isFinite(t) && Number.isFinite(m)) form.elements.TotalCharges.value = (t * m).toFixed(2);
}

const amount = (v) => (v.trim() !== '' && Number.isFinite(Number(v)) && Number(v) >= 0) || 'Enter an amount of 0 or more.';
const RULES = {
    tenure: (v) => /^\d+$/.test(v.trim()) || 'Enter whole months, 0 or more.',
    MonthlyCharges: amount,
    TotalCharges: amount,
};
function validate() {
    let first = null;
    for (const [name, rule] of Object.entries(RULES)) {
        const el = form.elements[name], result = rule(el.value), ok = result === true;
        el.setAttribute('aria-invalid', String(!ok));
        $(`err-${name}`).textContent = ok ? '' : result;
        if (!ok && !first) first = el;
    }
    first?.focus();
    return !first;
}

function collect() {
    const data = {};
    for (const el of form.elements) { // includes disabled selects, which FormData would drop
        if (!el.name || (el.type === 'radio' && !el.checked)) continue;
        data[el.name] = el.value;
    }
    data.SeniorCitizen = parseInt(data.SeniorCitizen, 10);
    data.tenure = parseInt(data.tenure, 10);
    data.MonthlyCharges = parseFloat(data.MonthlyCharges);
    return data; // TotalCharges stays a string, as before
}

function setState(state) {
    panel.dataset.state = state;
    panel.setAttribute('aria-busy', String(state === 'loading'));
    $('submitBtn').disabled = state === 'loading';
    $('submitBtn').classList.toggle('loading', state === 'loading');
    $('submitBtn').textContent = state === 'loading' ? 'Predicting' : 'Predict churn';
}

function showResult(r) {
    const p = Math.min(1, Math.max(0, Number(r.churn_probability)));
    const t = Number.isFinite(Number(r.threshold)) ? Number(r.threshold) : 0.5;
    const band = riskBand(p, t);
    panel.dataset.tone = band.tone;
    $('arc').style.strokeDashoffset = String(100 - p * 100);
    $('pct').textContent = (p * 100).toFixed(1) + '%';
    $('verdict').textContent = r.churn_prediction === 1 ? 'Likely to churn' : 'Likely to stay';
    $('band').textContent = band.label;
    $('band').hidden = false;
    $('detail').textContent = `Estimated probability of leaving. Flagged as churn at ${(t * 100).toFixed(0)}% or above.`;
    $('error').hidden = true;
    if (matchMedia('(max-width: 880px)').matches) {
        panel.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'nearest' });
    }
}

function showError(msg) {
    $('error').textContent = msg;
    $('error').hidden = false;
}

function clearResult() {
    panel.dataset.tone = '';
    $('arc').style.strokeDashoffset = '100';
    $('pct').textContent = '\u2013';
    $('verdict').textContent = 'No prediction yet';
    $('band').hidden = true;
    $('detail').textContent = 'Fill in the customer profile, then run the prediction.';
    $('error').hidden = true;
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!validate()) return;
    setState('loading');
    $('error').hidden = true;
    try {
        const res = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(collect()),
        });
        if (!res.ok) throw new Error(`The server responded with ${res.status}. Check the inputs and the API logs.`);
        showResult(await res.json());
        setState('done');
    } catch (err) {
        console.error(err);
        setState('error');
        showError(err instanceof TypeError
            ? "Couldn't reach the prediction service. Check that the API is running and try again."
            : `Prediction failed. ${err.message}`);
    }
});

form.addEventListener('input', (e) => {
    if (e.target.name === 'TotalCharges') totalTouched = true;
    if (e.target.name === 'tenure' || e.target.name === 'MonthlyCharges') syncTotal();
    if (e.target.name in RULES) { e.target.removeAttribute('aria-invalid'); $(`err-${e.target.name}`).textContent = ''; }
});
form.addEventListener('change', syncDependencies);
form.addEventListener('reset', () => {
    setTimeout(() => {
        totalTouched = false;
        Object.keys(RULES).forEach((n) => { form.elements[n].removeAttribute('aria-invalid'); $(`err-${n}`).textContent = ''; });
        syncDependencies();
        clearResult();
        setState('empty');
    }, 0);
});

syncDependencies();
