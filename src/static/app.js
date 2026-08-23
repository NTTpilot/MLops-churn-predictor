document.getElementById('churnForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // Convert numeric fields
    data.SeniorCitizen = parseInt(data.SeniorCitizen);
    data.tenure = parseInt(data.tenure);
    data.MonthlyCharges = parseFloat(data.MonthlyCharges);

    const resultDiv = document.getElementById('result');
    const probValue = document.getElementById('probValue');
    const predValue = document.getElementById('predValue');

    resultDiv.classList.add('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        // Show result
        probValue.textContent = (result.churn_probability * 100).toFixed(1) + '%';
        predValue.textContent = result.churn_label;

        if (result.churn_prediction === 1) {
            resultDiv.className = 'mt-8 p-6 rounded-xl border-2 bg-red-50 border-red-300';
            predValue.className = 'text-2xl font-bold text-red-600';
        } else {
            resultDiv.className = 'mt-8 p-6 rounded-xl border-2 bg-green-50 border-green-300';
            predValue.className = 'text-2xl font-bold text-green-600';
        }

        resultDiv.classList.remove('hidden');

    } catch (err) {
        alert('Error: ' + err.message);
        console.error(err);
    }
});