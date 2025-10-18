document.addEventListener('DOMContentLoaded', function() {
    // Add action buttons
    
    // Add logout button to the bottom of the sidebar
    const logoutContainer = document.querySelector('.logout-container');
    if (logoutContainer) {
        const logoutButton = document.createElement('button');
        logoutButton.textContent = 'Выход';
        logoutButton.className = 'action-button';
        logoutButton.style.marginTop = 'auto';
        logoutButton.style.marginBottom = '20px';
        logoutContainer.appendChild(logoutButton);
        
        logoutButton.addEventListener('click', async function() {
            try {
                // Clear the session token cookie by setting it to expire
                document.cookie = "session_token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
                
                // Redirect to login page
                window.location.href = '/login.html';
            } catch (error) {
                console.error('Ошибка выхода:', error);
                alert('Произошла ошибка при выходе. Пожалуйста, попробуйте снова.');
            }
        });
    }
});
