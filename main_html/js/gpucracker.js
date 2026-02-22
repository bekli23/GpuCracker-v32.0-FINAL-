/**
 * GpuCracker - JavaScript Demo Functionality
 */

document.addEventListener('DOMContentLoaded', () => {
    // Mode selection
    const modeCards = document.querySelectorAll('.mode-card');
    modeCards.forEach(card => {
        card.addEventListener('click', () => {
            modeCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            
            const mode = card.dataset.mode;
            console.log('Selected mode:', mode);
        });
    });

    // Puzzle selection
    const puzzleOptions = document.querySelectorAll('.puzzle-option');
    puzzleOptions.forEach(option => {
        option.addEventListener('click', () => {
            puzzleOptions.forEach(o => o.classList.remove('selected'));
            option.classList.add('selected');
            
            const puzzleNum = option.querySelector('.puzzle-num').textContent;
            updateTerminal(`[PUZZLE] Selected: ${puzzleNum}`);
        });
    });

    // GPU card selection
    const gpuCards = document.querySelectorAll('.gpu-card');
    gpuCards.forEach(card => {
        card.addEventListener('click', () => {
            gpuCards.forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
        });
    });

    // Start/Stop cracking
    const startBtn = document.getElementById('startCracking');
    const stopBtn = document.getElementById('stopCracking');
    const terminalOutput = document.getElementById('terminalOutput');
    const statusBadge = document.getElementById('statusBadge');
    const searchBar = document.getElementById('searchBar');
    const searchProgress = document.getElementById('searchProgress');
    const searchSpeed = document.getElementById('searchSpeed');
    const keysChecked = document.getElementById('keysChecked');

    let isRunning = false;
    let progress = 0;
    let checked = 0;

    function updateTerminal(message, type = 'info') {
        const line = document.createElement('div');
        line.className = `terminal-line ${type}`;
        line.textContent = message;
        terminalOutput.appendChild(line);
        terminalOutput.scrollTop = terminalOutput.scrollHeight;
    }

    startBtn?.addEventListener('click', () => {
        isRunning = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusBadge.innerHTML = '<i class="fas fa-circle"></i> RUNNING';
        statusBadge.className = 'status-badge status-success';
        
        updateTerminal('[START] GPU kernel initialized', 'info');
        updateTerminal('[CUDA] Blocks: 256, Threads: 512', 'info');
        updateTerminal('[MEMORY] Allocated 2GB GPU memory', 'info');
        
        // Simulate progress
        const interval = setInterval(() => {
            if (!isRunning) {
                clearInterval(interval);
                return;
            }
            
            progress += 0.0000001;
            checked += 15000000;
            
            searchBar.style.width = Math.min(progress * 100, 100) + '%';
            searchProgress.textContent = progress.toFixed(10) + '%';
            searchSpeed.textContent = (140 + Math.floor(Math.random() * 20)) + ' Mkeys/s';
            keysChecked.textContent = checked.toLocaleString() + ' / 2^71';
            
            // Random log messages
            if (Math.random() > 0.95) {
                const messages = [
                    `[CHECK] Batch #${Math.floor(checked / 1000000)} completed`,
                    `[HASH] Computing RIPEMD-160 for 131072 keys`,
                    `[BLOOM] Testing against filter layer 1`,
                    `[GPU] Temperature: ${62 + Math.floor(Math.random() * 8)}°C`,
                ];
                updateTerminal(messages[Math.floor(Math.random() * messages.length)]);
            }
        }, 100);
    });

    stopBtn?.addEventListener('click', () => {
        isRunning = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        statusBadge.innerHTML = '<i class="fas fa-circle"></i> STOPPED';
        statusBadge.className = 'status-badge status-pending';
        updateTerminal('[STOP] User requested stop', 'error');
        updateTerminal('[SAVE] Progress saved to checkpoint.dat', 'info');
    });

    // Save output
    document.getElementById('saveOutput')?.addEventListener('click', () => {
        alert('Log saved to gpucracker_log_2025.txt');
    });
});
