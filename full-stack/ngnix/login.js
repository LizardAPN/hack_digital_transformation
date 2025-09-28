document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.querySelector('.form');
    
    if (loginForm) {
        loginForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const login = document.getElementById('login').value;
            const password = document.getElementById('password').value;
            const remember = document.querySelector('input[name="remember"]').checked;
            
            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        name: login,
                        password: password
                    })
                });
                
                if (response.ok) {
                    // Login successful, redirect to workbench
                    window.location.href = '/workbench.html';
                } else {
                    // Handle login error
                    const errorData = await response.json();
                    alert('Login failed: ' + (errorData.detail || 'Unknown error'));
                }
            } catch (error) {
                console.error('Login error:', error);
                alert('An error occurred during login. Please try again.');
            }
        });
    }
});