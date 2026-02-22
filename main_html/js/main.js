/*
 * Main JavaScript for Bitcoin Recovery Suite
 * Handles animations, interactions, and UI effects
 */

// ============================================
// NAVIGATION
// ============================================
const navbar = document.querySelector('.navbar');
const hamburger = document.querySelector('.hamburger');
const navMenu = document.querySelector('.nav-menu');

// Navbar scroll effect
window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// Mobile menu toggle
if (hamburger) {
    hamburger.addEventListener('click', () => {
        navMenu.classList.toggle('active');
    });
}

// ============================================
// ANIMATED COUNTERS
// ============================================
function animateCounter(element, target, duration = 2000) {
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;
    
    const updateCounter = () => {
        current += increment;
        if (current < target) {
            element.textContent = Math.floor(current);
            requestAnimationFrame(updateCounter);
        } else {
            element.textContent = target;
        }
    };
    
    updateCounter();
}

// Intersection Observer for counters
const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const target = parseInt(entry.target.dataset.target);
            animateCounter(entry.target, target);
            counterObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.5 });

document.querySelectorAll('.stat-number').forEach(counter => {
    counterObserver.observe(counter);
});

// ============================================
// SCROLL REVEAL ANIMATIONS
// ============================================
const revealElements = document.querySelectorAll('.feature-card, .puzzle-card');

const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
            revealObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.1 });

revealElements.forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    revealObserver.observe(el);
});

// ============================================
// TERMINAL TYPING EFFECT
// ============================================
class TerminalTyper {
    constructor(element, text, speed = 50) {
        this.element = element;
        this.text = text;
        this.speed = speed;
        this.index = 0;
        this.isTyping = false;
    }
    
    type() {
        if (this.index < this.text.length) {
            this.element.textContent += this.text.charAt(this.index);
            this.index++;
            setTimeout(() => this.type(), this.speed);
        }
    }
    
    start() {
        this.element.textContent = '';
        this.type();
    }
}

// ============================================
// GPU ACCELERATION INDICATOR
// ============================================
function createGpuIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'gpu-indicator';
    indicator.innerHTML = `
        <div class="gpu-bar"></div>
        <div class="gpu-bar"></div>
        <div class="gpu-bar"></div>
        <div class="gpu-bar"></div>
        <span>GPU Active</span>
    `;
    indicator.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: rgba(10, 10, 15, 0.9);
        border: 1px solid rgba(247, 147, 26, 0.3);
        border-radius: 12px;
        padding: 10px 15px;
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #f7931a;
        z-index: 9999;
        backdrop-filter: blur(10px);
    `;
    
    const style = document.createElement('style');
    style.textContent = `
        .gpu-bar {
            width: 4px;
            height: 20px;
            background: rgba(247, 147, 26, 0.3);
            border-radius: 2px;
            animation: gpuPulse 1s infinite ease-in-out;
        }
        .gpu-bar:nth-child(2) { animation-delay: 0.1s; }
        .gpu-bar:nth-child(3) { animation-delay: 0.2s; }
        .gpu-bar:nth-child(4) { animation-delay: 0.3s; }
        
        @keyframes gpuPulse {
            0%, 100% { height: 10px; background: rgba(247, 147, 26, 0.3); }
            50% { height: 25px; background: rgba(247, 147, 26, 1); }
        }
    `;
    document.head.appendChild(style);
    document.body.appendChild(indicator);
}

// ============================================
// PARTICLE EFFECT
// ============================================
class ParticleSystem {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        
        this.canvas.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 0;
            opacity: 0.3;
        `;
        
        document.body.appendChild(this.canvas);
        
        this.resize();
        window.addEventListener('resize', () => this.resize());
        
        this.createParticles();
        this.animate();
    }
    
    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }
    
    createParticles() {
        const particleCount = 50;
        for (let i = 0; i < particleCount; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                size: Math.random() * 3 + 1,
                speedX: (Math.random() - 0.5) * 0.5,
                speedY: (Math.random() - 0.5) * 0.5,
                opacity: Math.random() * 0.5 + 0.2
            });
        }
    }
    
    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.particles.forEach(p => {
            p.x += p.speedX;
            p.y += p.speedY;
            
            if (p.x < 0) p.x = this.canvas.width;
            if (p.x > this.canvas.width) p.x = 0;
            if (p.y < 0) p.y = this.canvas.height;
            if (p.y > this.canvas.height) p.y = 0;
            
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            this.ctx.fillStyle = `rgba(247, 147, 26, ${p.opacity})`;
            this.ctx.fill();
        });
        
        requestAnimationFrame(() => this.animate());
    }
}

// ============================================
// TOOLTIP SYSTEM
// ============================================
function createTooltip(element, text) {
    const tooltip = document.createElement('div');
    tooltip.className = 'custom-tooltip';
    tooltip.textContent = text;
    tooltip.style.cssText = `
        position: absolute;
        background: rgba(10, 10, 15, 0.95);
        border: 1px solid rgba(247, 147, 26, 0.3);
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 12px;
        color: #fff;
        z-index: 10000;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.3s;
        max-width: 250px;
        backdrop-filter: blur(10px);
    `;
    
    document.body.appendChild(tooltip);
    
    element.addEventListener('mouseenter', (e) => {
        tooltip.style.opacity = '1';
        updateTooltipPosition(e);
    });
    
    element.addEventListener('mousemove', updateTooltipPosition);
    
    element.addEventListener('mouseleave', () => {
        tooltip.style.opacity = '0';
    });
    
    function updateTooltipPosition(e) {
        const x = e.clientX + 15;
        const y = e.clientY + 15;
        tooltip.style.left = x + 'px';
        tooltip.style.top = y + 'px';
    }
}

// ============================================
// KEYBOARD SHORTCUTS
// ============================================
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K to focus search (if exists)
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('.search-input');
        if (searchInput) searchInput.focus();
    }
    
    // Escape to close mobile menu
    if (e.key === 'Escape') {
        if (navMenu) navMenu.classList.remove('active');
    }
});

// ============================================
// LOADING SCREEN
// ============================================
function showLoadingScreen() {
    const loader = document.createElement('div');
    loader.className = 'loading-screen';
    loader.innerHTML = `
        <div class="loader-content">
            <div class="loader-logo">
                <i class="fas fa-key"></i>
            </div>
            <div class="loader-text">Loading Tools...</div>
            <div class="loader-bar">
                <div class="loader-progress"></div>
            </div>
        </div>
    `;
    loader.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: #0a0a0f;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 99999;
        transition: opacity 0.5s ease;
    `;
    
    const style = document.createElement('style');
    style.textContent = `
        .loader-content {
            text-align: center;
        }
        .loader-logo {
            font-size: 4rem;
            color: #f7931a;
            margin-bottom: 1rem;
            animation: pulse 1.5s infinite;
        }
        .loader-text {
            color: #fff;
            font-size: 1.25rem;
            margin-bottom: 1.5rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .loader-bar {
            width: 300px;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
        }
        .loader-progress {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #f7931a, #ff006e);
            animation: loadProgress 2s ease-in-out forwards;
        }
        @keyframes loadProgress {
            0% { width: 0%; }
            100% { width: 100%; }
        }
    `;
    document.head.appendChild(style);
    document.body.appendChild(loader);
    
    setTimeout(() => {
        loader.style.opacity = '0';
        setTimeout(() => loader.remove(), 500);
    }, 2000);
}

// ============================================
// INITIALIZATION
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    // Show loading screen on first visit
    if (!sessionStorage.getItem('loaded')) {
        showLoadingScreen();
        sessionStorage.setItem('loaded', 'true');
    }
    
    // Initialize particle system
    new ParticleSystem();
    
    // Add GPU indicator
    createGpuIndicator();
    
    // Add tooltips to feature cards
    document.querySelectorAll('.feature-card').forEach(card => {
        const title = card.querySelector('h3')?.textContent;
        if (title) {
            createTooltip(card, `Click to open ${title}`);
        }
    });
    
    // Add hover effect to puzzle cards
    document.querySelectorAll('.puzzle-card').forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-10px) scale(1.02)';
        });
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0) scale(1)';
        });
    });
    
    // Console Easter Egg
    console.log('%c🔐 Bitcoin Recovery Suite', 'font-size: 24px; font-weight: bold; color: #f7931a;');
    console.log('%cAdvanced Cryptographic Tools', 'font-size: 14px; color: #00d4ff;');
    console.log('%c⚠️ For educational purposes only', 'font-size: 12px; color: #ff006e;');
});

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});
