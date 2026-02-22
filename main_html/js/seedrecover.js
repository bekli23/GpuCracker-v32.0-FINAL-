/**
 * SeedRecover - JavaScript Demo Functionality
 */

document.addEventListener('DOMContentLoaded', () => {
    // Toggle buttons
    const toggleBtns = document.querySelectorAll('.btn-toggle');
    toggleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const parent = btn.parentElement;
            parent.querySelectorAll('.btn-toggle').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Start/Stop buttons
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const progressBar = document.getElementById('progressBar');
    const progressPercent = document.getElementById('progressPercent');
    const elapsedTime = document.getElementById('elapsedTime');
    const speed = document.getElementById('speed');
    const combinations = document.getElementById('combinations');
    const resultsTable = document.getElementById('resultsTable').querySelector('tbody');

    let isRunning = false;
    let progress = 0;
    let timer = null;
    let seconds = 0;

    startBtn?.addEventListener('click', () => {
        isRunning = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        
        // Clear results
        resultsTable.innerHTML = `
            <tr class="empty-row">
                <td colspan="5" class="empty-message">
                    <i class="fas fa-spinner fa-spin"></i>
                    <p>Searching... Please wait.</p>
                </td>
            </tr>
        `;

        // Simulate progress
        timer = setInterval(() => {
            if (!isRunning) return;
            
            progress += Math.random() * 2;
            if (progress >= 100) progress = 100;
            
            progressBar.style.width = progress + '%';
            progressPercent.textContent = progress.toFixed(1) + '%';
            
            seconds++;
            const hrs = Math.floor(seconds / 3600).toString().padStart(2, '0');
            const mins = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
            const secs = (seconds % 60).toString().padStart(2, '0');
            elapsedTime.textContent = `${hrs}:${mins}:${secs}`;
            
            const speedVal = 400000 + Math.floor(Math.random() * 100000);
            speed.textContent = speedVal.toLocaleString() + ' phrases/sec';
            
            const total = 4194304;
            const current = Math.floor(total * (progress / 100));
            combinations.textContent = current.toLocaleString() + ' / ' + total.toLocaleString();
            
            // Simulate finding a result at 35%
            if (progress >= 35 && progress < 36 && resultsTable.querySelector('.empty-row')) {
                addResult({
                    id: 1,
                    phrase: 'abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon',
                    address: '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
                    status: 'VALID'
                });
            }
            
            if (progress >= 100) {
                stopSearch();
            }
        }, 100);
    });

    stopBtn?.addEventListener('click', stopSearch);

    function stopSearch() {
        isRunning = false;
        clearInterval(timer);
        startBtn.disabled = false;
        stopBtn.disabled = true;
        
        if (resultsTable.querySelector('.empty-row')) {
            resultsTable.innerHTML = `
                <tr class="empty-row">
                    <td colspan="5" class="empty-message">
                        <i class="fas fa-inbox"></i>
                        <p>Search stopped. No additional results found.</p>
                    </td>
                </tr>
            `;
        }
    }

    function addResult(data) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${data.id}</td>
            <td class="mono">${data.phrase}</td>
            <td class="mono">${data.address}</td>
            <td><span class="status-badge status-success"><i class="fas fa-check"></i> ${data.status}</span></td>
            <td>
                <button class="btn btn-sm btn-secondary"><i class="fas fa-copy"></i></button>
                <button class="btn btn-sm btn-secondary"><i class="fas fa-qrcode"></i></button>
            </td>
        `;
        resultsTable.innerHTML = '';
        resultsTable.appendChild(row);
        
        document.getElementById('resultCount').textContent = data.id;
    }

    // Export button
    document.getElementById('exportBtn')?.addEventListener('click', () => {
        alert('Export functionality would save results to a file.');
    });
});
