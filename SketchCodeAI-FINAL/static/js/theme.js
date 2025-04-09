/**
 * Theme management script for Sketch to Code
 * Handles dark/light mode switching and persistence
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize theme from localStorage or use default (dark)
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);
    
    // Theme toggle button event listener
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    themeToggleBtn.addEventListener('click', toggleTheme);
    
    // Function to toggle between dark and light themes
    function toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-bs-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        setTheme(newTheme);
        
        // Save preference to localStorage
        localStorage.setItem('theme', newTheme);
    }
    
    // Function to apply theme to the document
    function setTheme(theme) {
        // Set the theme attribute on the HTML element
        document.documentElement.setAttribute('data-bs-theme', theme);
        
        // Update the theme toggle button
        const themeIcon = document.getElementById('themeIcon');
        const themeText = document.getElementById('themeText');
        
        // Switch Prism.js theme for code highlighting
        const prismDarkTheme = document.getElementById('prism-dark-theme');
        const prismLightTheme = document.getElementById('prism-light-theme');
        
        if (theme === 'dark') {
            themeIcon.className = 'fas fa-moon me-1';
            themeText.textContent = 'Dark';
            
            // Update navbar and footer for dark theme
            document.querySelector('.navbar').classList.add('navbar-dark');
            document.querySelector('.navbar').classList.add('bg-dark');
            document.querySelector('.navbar').classList.remove('navbar-light');
            document.querySelector('.navbar').classList.remove('bg-light');
            
            document.querySelector('.footer').classList.add('bg-dark');
            document.querySelector('.footer').classList.remove('bg-light');
            
            // Enable dark theme for Prism.js
            prismDarkTheme.disabled = false;
            prismLightTheme.disabled = true;
            
            // Force re-highlight of code blocks
            if (window.Prism) {
                setTimeout(() => {
                    Prism.highlightAll();
                }, 100);
            }
        } else {
            themeIcon.className = 'fas fa-sun me-1';
            themeText.textContent = 'Light';
            
            // Update navbar and footer for light theme
            document.querySelector('.navbar').classList.remove('navbar-dark');
            document.querySelector('.navbar').classList.remove('bg-dark');
            document.querySelector('.navbar').classList.add('navbar-light');
            document.querySelector('.navbar').classList.add('bg-light');
            
            document.querySelector('.footer').classList.remove('bg-dark');
            document.querySelector('.footer').classList.add('bg-light');
            
            // Enable light theme for Prism.js
            prismDarkTheme.disabled = true;
            prismLightTheme.disabled = false;
            
            // Force re-highlight of code blocks
            if (window.Prism) {
                setTimeout(() => {
                    Prism.highlightAll();
                }, 100);
            }
        }
    }
});