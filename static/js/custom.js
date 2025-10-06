// File: static/js/custom.js
// Custom JavaScript for FE-AI Dashboard

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 FE-AI Dashboard Initialized');
    
    // Add smooth scrolling
    initSmoothScroll();
    
    // Add interactive elements
    initInteractiveElements();
    
    // Initialize tooltips
    initTooltips();
    
    // Add keyboard shortcuts
    initKeyboardShortcuts();
});

// Smooth scrolling for anchor links
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Interactive card animations
function initInteractiveElements() {
    const cards = document.querySelectorAll('.feature-card, .metric-card');
    
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
}

// Initialize tooltips
function initTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', function() {
            showTooltip(this, this.getAttribute('data-tooltip'));
        });
        
        element.addEventListener('mouseleave', function() {
            hideTooltip();
        });
    });
}

function showTooltip(element, text) {
    const tooltip = document.createElement('div');
    tooltip.className = 'custom-tooltip';
    tooltip.textContent = text;
    tooltip.style.cssText = `
        position: absolute;
        background: #2c3e50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-size: 14px;
        z-index: 1000;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    `;
    
    document.body.appendChild(tooltip);
    
    const rect = element.getBoundingClientRect();
    tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
    tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + 'px';
}

function hideTooltip() {
    const tooltips = document.querySelectorAll('.custom-tooltip');
    tooltips.forEach(tooltip => tooltip.remove());
}

// Keyboard shortcuts
function initKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl+K: Quick search
        if (e.ctrlKey && e.key === 'k') {
            e.preventDefault();
            focusSearchBar();
        }
        
        // Ctrl+U: Upload data
        if (e.ctrlKey && e.key === 'u') {
            e.preventDefault();
            navigateToUpload();
        }
        
        // Ctrl+R: Run analysis
        if (e.ctrlKey && e.key === 'r') {
            e.preventDefault();
            runAnalysis();
        }
    });
}

function focusSearchBar() {
    const searchInput = document.querySelector('input[type="text"]');
    if (searchInput) {
        searchInput.focus();
    }
}

function navigateToUpload() {
    console.log('Navigate to upload page');
    // Implementation depends on your routing
}

function runAnalysis() {
    console.log('Run analysis triggered');
    // Implementation depends on your analysis workflow
}

// File upload drag and drop enhancement
function initDragAndDrop() {
    const uploadZone = document.querySelector('.upload-zone');
    
    if (uploadZone) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadZone.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight(e) {
            uploadZone.classList.add('drag-active');
        }
        
        function unhighlight(e) {
            uploadZone.classList.remove('drag-active');
        }
        
        uploadZone.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files);
        }
        
        function handleFiles(files) {
            console.log('Files dropped:', files);
            // Process files
        }
    }
}

// Progress animation
function animateProgress(element, targetValue, duration = 1000) {
    const start = 0;
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const currentValue = start + (targetValue - start) * easeOutCubic(progress);
        element.style.width = currentValue + '%';
        element.textContent = Math.round(currentValue) + '%';
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
}

// Real-time notifications
function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        z-index: 10000;
        animation: slideIn 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    `;
    
    // Set background based on type
    const colors = {
        'info': 'linear-gradient(135deg, #3498db, #2980b9)',
        'success': 'linear-gradient(135deg, #2ecc71, #27ae60)',
        'warning': 'linear-gradient(135deg, #f39c12, #e67e22)',
        'error': 'linear-gradient(135deg, #e74c3c, #c0392b)'
    };
    notification.style.background = colors[type] || colors.info;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, duration);
}

// Export functions for global use
window.FE_AI = {
    showNotification,
    animateProgress,
    initDragAndDrop
};

console.log('✅ FE-AI Custom JS Loaded');