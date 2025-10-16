document.addEventListener('DOMContentLoaded', function() {
    const registerForm = document.querySelector('.form');
    
    if (registerForm) {
        registerForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const name = document.getElementById('name').value;
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            const confirmPassword = document.getElementById('confirm-password').value;
            
            // Check if passwords match
            if (password !== confirmPassword) {
                alert('Пароли не совпадают');
                return;
            }
            
            try {
                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        name: name,
                        email: email,
                        password: password
                    })
                });
                
                if (response.ok) {
                    // Registration successful, redirect to workbench
                    window.location.href = '/workbench.html';
                } else {
                    // Handle registration error
                    const errorData = await response.json();
                    alert('Ошибка регистрации: ' + (errorData.detail || 'Неизвестная ошибка'));
                }
            } catch (error) {
                console.error('Ошибка регистрации:', error);
                alert('Произошла ошибка при регистрации. Пожалуйста, попробуйте снова.');
            }
        });
    }
});
