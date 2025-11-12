document.getElementById('predict-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const data = Object.fromEntries(formData);
    // Always request all models so we can display comparative predictions
    data.model = 'all';

    try {
        const btn = document.getElementById('predict-btn');
        const spinner = document.getElementById('spinner');
        btn.disabled = true;
        spinner.classList.add('show-spinner');

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        if (response.ok) {
            const container = document.getElementById('result');
            container.innerHTML = '';
            const ensembleSummary = document.getElementById('ensemble-summary');
            const ensembleLabel = ensembleSummary.querySelector('.ensemble-label');
            const ensembleProb = ensembleSummary.querySelector('.ensemble-prob');
            // Hide the agreement/probability small note per user request
            ensembleProb.hidden = true;
            const detailsToggle = document.getElementById('details-toggle');
            const detailsDropdown = document.getElementById('details-dropdown');

            if (result.results) {
                const modelColors = {
                    dt: 'linear-gradient(90deg,#7f3f98,#9b6cff)',
                    svm: 'linear-gradient(90deg,#00a99d,#5eead4)',
                    rf: 'linear-gradient(90deg,#0b74de,#5ab0ff)',
                    xgb: 'linear-gradient(90deg,#f97316,#ffd166)',
                    ann: 'linear-gradient(90deg,#ff6b6b,#ff9a9a)'
                };

                // Build per-model cards but keep them inside the dropdown
                const entries = Object.entries(result.results);
                const preds = [];
                const probs = [];
                entries.forEach(([model, res], idx) => {
                    const card = document.createElement('div');
                    card.className = 'card';
                    const accent = document.createElement('div');
                    accent.className = 'card-top';
                    accent.style.background = modelColors[model] || 'linear-gradient(90deg,#e2e8f0,#cbd5e1)';
                    card.appendChild(accent);
                    if (res.error) {
                        const inner = document.createElement('div');
                        inner.innerHTML = `<h4>${model.toUpperCase()}</h4><p class="error">Error: ${res.error}</p>`;
                        card.appendChild(inner);
                        card.classList.add('bad');
                    } else {
                        const label = res.prediction === 1 ? 'Disease' : 'No Disease';
                        const inner = document.createElement('div');
                        inner.innerHTML = `<h4>${model.toUpperCase()}</h4><p>${label}</p><p>${(res.probability*100).toFixed(2)}% probability</p>`;
                        card.appendChild(inner);
                        if (res.prediction === 1) card.classList.add('bad'); else card.classList.add('good');
                        preds.push(res.prediction);
                        probs.push(res.probability);
                    }
                    container.appendChild(card);
                    setTimeout(() => card.classList.add('show'), idx * 80);
                });

                // Compute majority vote
                const counts = {};
                preds.forEach(p => { counts[p] = (counts[p] || 0) + 1 });
                const total = preds.length;
                let majorityLabel = null;
                let majorityCount = 0;
                Object.keys(counts).forEach(k => { if (counts[k] > majorityCount) { majorityCount = counts[k]; majorityLabel = parseInt(k) } });

                const labelText = majorityLabel === 1 ? 'Disease' : 'No Disease';
                const agreement = (majorityCount / total) * 100;

                // Average probability among models that voted for the majority label
                let probSum = 0; let probCount = 0;
                entries.forEach(([m, r]) => { if (!r.error && r.prediction === majorityLabel) { probSum += r.probability; probCount += 1 } });
                const avgProb = probCount ? (probSum / probCount) : (probs.length ? (probs.reduce((a,b)=>a+b,0)/probs.length) : 0);

                // User-requested compact narrative messages
                const probStr = (avgProb*100).toFixed(1) + '%';
                if (majorityLabel === 1) {
                    // Example: Our analysis indicates a potential risk of heart disease (69.9%). All risks are possible.
                    ensembleLabel.textContent = `Our analysis indicates a potential risk of heart disease (${probStr}). All risks are possible.`;
                } else {
                    // Example: Our analysis indicates a lower risk of heart disease (28.0%). Some risks may still exist.
                    ensembleLabel.textContent = `Our analysis indicates a lower risk of heart disease (${probStr}). Some risks may still exist.`;
                }
                // agreement/probability note hidden by user preference
                // ensembleProb.textContent intentionally omitted
                ensembleSummary.hidden = false;
                detailsDropdown.hidden = true;
                detailsToggle.setAttribute('aria-expanded','false');

                // Toggle behavior
                detailsToggle.onclick = () => {
                    const expanded = detailsToggle.getAttribute('aria-expanded') === 'true';
                    detailsToggle.setAttribute('aria-expanded', (!expanded).toString());
                    detailsDropdown.hidden = expanded;
                    detailsToggle.textContent = expanded ? 'Show model details ▾' : 'Hide model details ▴';
                };

            } else {
                // Single-model fallback: same as before but show ensemble area for consistency
                const card = document.createElement('div');
                card.className = 'card';
                const accent = document.createElement('div');
                accent.className = 'card-top';
                accent.style.background = 'linear-gradient(90deg,#cbd5e1,#e2e8f0)';
                card.appendChild(accent);
                const label = result.prediction === 1 ? 'Disease' : 'No Disease';
                card.innerHTML = `<h4>${result.model.toUpperCase()}</h4><p>${label}</p><p>${(result.probability*100).toFixed(2)}% probability</p>`;
                if (result.prediction === 1) card.classList.add('bad'); else card.classList.add('good');
                container.appendChild(card);
                setTimeout(() => card.classList.add('show'), 60);

                // Single-model narrative
                const pStr = (result.probability*100).toFixed(1) + '%';
                if (result.prediction === 1) {
                    ensembleLabel.textContent = `Our analysis indicates a potential risk of heart disease (${pStr}). We recommend consulting a healthcare professional.`;
                } else {
                    ensembleLabel.textContent = `Our analysis indicates a lower risk of heart disease (${pStr}). Continue maintaining healthy habits.`;
                }
                // agreement/probability note hidden by user preference
                ensembleSummary.hidden = false;
                document.getElementById('details-dropdown').hidden = true;
                document.getElementById('details-toggle').style.display = 'none';
            }
        } else {
            document.getElementById('result').innerHTML = `<p>Error: ${result.error}</p>`;
            document.getElementById('result').style.background = '#f8d7da';
        }
    } catch (error) {
        document.getElementById('result').innerHTML = `<p>Error: ${error.message}</p>`;
        document.getElementById('result').style.background = '#f8d7da';
    } finally {
        const btn = document.getElementById('predict-btn');
        const spinner = document.getElementById('spinner');
        btn.disabled = false;
        spinner.classList.remove('show-spinner');
    }
});
