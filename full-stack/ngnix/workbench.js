// Workbench JavaScript functionality

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const uploadButton = document.getElementById('upload-button');
    const photoUploadInput = document.getElementById('photo-upload');
    const photoGrid = document.getElementById('photo-grid');
    const uploadStatus = document.getElementById('upload-status');
    const addPhotoCard = document.getElementById('add-photo-card');
    
    // Modal elements
    const modal = document.getElementById('photo-modal');
    const modalImage = document.getElementById('modal-image');
    const modalCaption = document.getElementById('modal-caption');
    const closeModal = document.getElementsByClassName('close')[0];
    
    // Event listener for upload button
    uploadButton.addEventListener('click', function() {
        photoUploadInput.click();
    });
    
    // Event listener for file selection
    photoUploadInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            uploadPhoto(file);
        }
    });
    
    // Event listener for add photo card
    addPhotoCard.addEventListener('click', function() {
        photoUploadInput.click();
    });
    
    // Event listener for modal close button
    closeModal.addEventListener('click', function() {
        modal.style.display = 'none';
    });
    
    // Event listener for clicking outside modal to close
    window.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
    
    // Function to upload photo
    function uploadPhoto(file) {
        uploadStatus.textContent = 'Uploading...';
        uploadStatus.className = 'upload-status uploading';
        
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/api/photo_upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error('Upload failed');
            }
        })
        .then(data => {
            uploadStatus.textContent = 'Upload successful!';
            uploadStatus.className = 'upload-status success';
            loadPhotos(); // Refresh photo grid
            photoUploadInput.value = ''; // Reset file input to allow re-upload of same file
            setTimeout(() => {
                uploadStatus.textContent = '';
                uploadStatus.className = 'upload-status';
            }, 3000);
        })
        .catch(error => {
            uploadStatus.textContent = 'Upload failed. Please try again.';
            uploadStatus.className = 'upload-status error';
            console.error('Upload error:', error);
            photoUploadInput.value = ''; // Reset file input to allow re-upload of same file
        });
    }
    
    // Function to load photos
    function loadPhotos() {
        fetch('/api/photos', {
            method: 'GET',
            credentials: 'include'
        })
        .then(response => response.json())
        .then(data => {
            displayPhotos(data.photos);
        })
        .catch(error => {
            console.error('Error loading photos:', error);
        });
    }
    
    // Function to display photos in grid
    function displayPhotos(photos) {
        // Clear existing photos except the add photo card
        photoGrid.innerHTML = '';
        photoGrid.appendChild(addPhotoCard);
        
        // Add photos to grid
        photos.forEach(photo => {
            const photoCard = document.createElement('div');
            photoCard.className = 'photo-card';
            photoCard.innerHTML = `
                <img src="${photo.photo_url}" alt="${photo.created_at}" class="photo-thumbnail">
                <p class="photo-name">${photo.created_at}</p>
            `;
            
            // Add click event to open modal
            photoCard.addEventListener('click', function() {
                openPhotoModal(photo);
            });
            
            photoGrid.insertBefore(photoCard, addPhotoCard);
        });
    }
    
    // Function to open photo in modal
    function openPhotoModal(photo) {
        modalImage.src = photo.photo_url;
        modalCaption.innerHTML = `<h3>${photo.created_at}</h3><p>Uploaded: ${new Date(photo.created_at).toLocaleString()}</p>`;
        modal.style.display = 'block';
    }
    
    // Load photos when page loads
    loadPhotos();
});