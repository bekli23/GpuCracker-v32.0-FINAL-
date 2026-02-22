/**
 * AKM Tools - JavaScript Demo Functionality
 */

document.addEventListener('DOMContentLoaded', () => {
    // Profile card selection
    const profileCards = document.querySelectorAll('.profile-card');
    profileCards.forEach(card => {
        card.addEventListener('click', () => {
            profileCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
        });
    });

    // Filter tags
    const filterTags = document.querySelectorAll('.filter-tag');
    filterTags.forEach(tag => {
        tag.addEventListener('click', () => {
            if (tag.textContent === 'All') {
                filterTags.forEach(t => t.classList.remove('active'));
                tag.classList.add('active');
            } else {
                document.querySelector('.filter-tag').classList.remove('active');
                tag.classList.toggle('active');
            }
        });
    });

    // Convert button
    const convertBtn = document.getElementById('convertBtn');
    convertBtn?.addEventListener('click', () => {
        const phrase = document.getElementById('akmPhrase')?.value || 'abis acelasi acoperis adanc adapost';
        
        // Simulate conversion
        const words = phrase.split(' ');
        const hexValues = words.map((w, i) => {
            const hash = w.split('').reduce((a, b) => a + b.charCodeAt(0), 0);
            return (hash % 256).toString(16).padStart(2, '0');
        });
        
        const privKey = hexValues.join('') + '00'.repeat(16 - hexValues.length);
        
        const privKeyDisplay = document.getElementById('privKeyHex');
        const btcAddressDisplay = document.getElementById('btcAddress');
        
        // Animate the result
        privKeyDisplay.innerHTML = '<span class="typing"></span>';
        btcAddressDisplay.innerHTML = '<span class="typing"></span>';
        
        typeWriter(privKeyDisplay.querySelector('.typing'), privKey, 20, () => {
            const address = '1' + Array(33).fill(0).map(() => 
                '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'[Math.floor(Math.random() * 58)]
            ).join('');
            typeWriter(btcAddressDisplay.querySelector('.typing'), address, 10);
        });
    });

    function typeWriter(element, text, speed, callback) {
        let i = 0;
        element.textContent = '';
        
        function type() {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(type, speed);
            } else if (callback) {
                callback();
            }
        }
        
        type();
    }

    // Generate phrase button
    const generateBtn = document.getElementById('generatePhrase');
    generateBtn?.addEventListener('click', () => {
        const seed = document.getElementById('genSeed')?.value || Math.floor(Math.random() * 1000000);
        const length = parseInt(document.getElementById('genLength')?.value || 10);
        
        // Romanian wordlist simulation
        const words = [
            'abis', 'acelasi', 'acoperis', 'adanc', 'adapost', 'adorare', 'afectiune', 'aer',
            'ajun', 'albastru', 'alb', 'alge', 'altar', 'amintire', 'amurg', 'anotimp',
            'ape', 'arbori', 'arc', 'ardere', 'arma', 'arsura', 'artar', 'asemenea'
        ];
        
        const phrase = [];
        let seedNum = parseInt(seed) || Math.floor(Math.random() * 1000000);
        
        for (let i = 0; i < length; i++) {
            seedNum = (seedNum * 9301 + 49297) % 233280;
            const index = Math.floor((seedNum / 233280) * words.length);
            phrase.push(words[index]);
        }
        
        const display = document.getElementById('generatedPhrase');
        display.textContent = phrase.join(' ');
        display.style.color = 'var(--primary)';
    });

    // Random seed button
    document.getElementById('randomSeed')?.addEventListener('click', () => {
        const seedInput = document.getElementById('genSeed');
        if (seedInput) {
            seedInput.value = Math.floor(Math.random() * 1000000000);
        }
    });

    // Profile search
    const profileSearch = document.getElementById('profileSearch');
    profileSearch?.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        profileCards.forEach(card => {
            const text = card.textContent.toLowerCase();
            card.style.display = text.includes(query) ? '' : 'none';
        });
    });
});
