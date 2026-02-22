/**
 * Bloom Filter Builder - JavaScript Demo Functionality
 */

document.addEventListener('DOMContentLoaded', () => {
    // Tab switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(`tab-${tab}`).classList.add('active');
        });
    });

    // File drop zone
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('addressFile');

    dropZone?.addEventListener('click', () => fileInput?.click());

    dropZone?.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary)';
        dropZone.style.background = 'rgba(247, 147, 26, 0.1)';
    });

    dropZone?.addEventListener('dragleave', () => {
        dropZone.style.borderColor = '';
        dropZone.style.background = '';
    });

    dropZone?.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '';
        dropZone.style.background = '';
        handleFiles(e.dataTransfer.files);
    });

    fileInput?.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        const fileList = document.getElementById('fileList');
        fileList.innerHTML = '';
        
        let totalSize = 0;
        let totalLines = 0;

        Array.from(files).forEach(file => {
            totalSize += file.size;
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <i class="fas fa-file"></i>
                <span>${file.name}</span>
                <span class="file-size">${(file.size / 1024 / 1024).toFixed(2)} MB</span>
            `;
            fileList.appendChild(fileItem);
            
            // Simulate line counting
            totalLines += Math.floor(file.size / 35);
        });

        document.getElementById('bloomFileCount').textContent = files.length;
        document.getElementById('bloomSize').textContent = (totalSize / 1024 / 1024).toFixed(1) + ' MB';
        document.getElementById('bloomCapacity').textContent = totalLines.toLocaleString();
        document.getElementById('totalAddresses').textContent = totalLines.toLocaleString();
        document.getElementById('validAddresses').textContent = totalLines.toLocaleString();
    }

    // Build button
    const buildBtn = document.getElementById('buildBtn');
    const buildBar = document.getElementById('buildBar');
    const buildProgress = document.getElementById('buildProgress');
    const buildSteps = document.querySelectorAll('.build-step');

    buildBtn?.addEventListener('click', () => {
        let progress = 0;
        buildBtn.disabled = true;
        
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
                buildBtn.disabled = false;
                
                // Mark all steps complete
                buildSteps.forEach(step => {
                    step.classList.add('completed');
                    step.querySelector('i').className = 'fas fa-check-circle';
                });
            }
            
            buildBar.style.width = progress + '%';
            buildProgress.textContent = Math.floor(progress) + '%';
            
            // Update steps
            if (progress >= 25) {
                buildSteps[0].classList.add('completed');
                buildSteps[1].classList.add('active');
            }
            if (progress >= 50) {
                buildSteps[1].classList.add('completed');
                buildSteps[2].classList.add('active');
            }
            if (progress >= 75) {
                buildSteps[2].classList.add('completed');
                buildSteps[3].classList.add('active');
            }
        }, 200);
    });

    // Calculator
    const calcItems = document.getElementById('calcItems');
    const calcFPR = document.getElementById('calcFPR');
    
    function updateCalculation() {
        const n = parseInt(calcItems?.value || 1000000);
        const p = parseFloat(calcFPR?.value || 0.01);
        
        // Bloom filter calculations
        const m = Math.ceil(-(n * Math.log(p)) / (Math.log(2) ** 2));
        const k = Math.round((m / n) * Math.log(2));
        const bitsPerItem = m / n;
        
        document.getElementById('calcSize').textContent = (m / 8 / 1024 / 1024).toFixed(2) + ' MB';
        document.getElementById('calcHashes').textContent = k;
        document.getElementById('calcBits').textContent = bitsPerItem.toFixed(1);
    }

    calcItems?.addEventListener('input', updateCalculation);
    calcFPR?.addEventListener('change', updateCalculation);
});
