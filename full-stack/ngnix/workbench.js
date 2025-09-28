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
            setTimeout(() => {
                uploadStatus.textContent = '';
                uploadStatus.className = 'upload-status';
            }, 3000);
        })
        .catch(error => {
            uploadStatus.textContent = 'Upload failed. Please try again.';
            uploadStatus.className = 'upload-status error';
            console.error('Upload error:', error);
        });
    }
    
    // Function to load photos
    function loadPhotos() {
        // First load photos
        fetch('/api/photos', {
            method: 'GET',
            credentials: 'include'
        })
        .then(response => response.json())
        .then(data => {
            const photos = data.photos;
            // For each photo, try to get processing results
            const photoPromises = photos.map(photo => {
                return fetch(`/api/results/latest?limit=100`, {
                    method: 'GET',
                    credentials: 'include'
                })
                .then(response => response.json())
                .then(resultsData => {
                    // Find result for this photo
                    const photoResults = resultsData.results.filter(result => 
                        result.image_path === photo.photo_url
                    );
                    photo.processing_results = photoResults;
                    return photo;
                })
                .catch(error => {
                    console.error('Error loading processing results for photo:', photo.photo_url, error);
                    photo.processing_results = [];
                    return photo;
                });
            });
            
            // Wait for all photo data to be loaded
            Promise.all(photoPromises)
                .then(photosWithResults => {
                    displayPhotos(photosWithResults);
                })
                .catch(error => {
                    console.error('Error loading photos with results:', error);
                    // Fallback to displaying photos without results
                    displayPhotos(photos);
                });
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
        let captionHTML = `<h3>${photo.created_at}</h3><p>Uploaded: ${new Date(photo.created_at).toLocaleString()}</p>`;
        
        // Add processing results if available
        const processingResultsDiv = document.getElementById('processing-results');
        processingResultsDiv.innerHTML = '';
        
        if (photo.processing_results && photo.processing_results.length > 0) {
            // Sort results by processed_at timestamp (newest first)
            const sortedResults = photo.processing_results.sort((a, b) => 
                new Date(b.processed_at) - new Date(a.processed_at)
            );
            
            const latestResult = sortedResults[0];
            
            // Add processing results to caption
            captionHTML += '<div class="processing-info">';
            captionHTML += '<h4>Processing Results</h4>';
            
            // Display coordinates if available
            if (latestResult.coordinates && latestResult.coordinates.lat && latestResult.coordinates.lon) {
                captionHTML += `<p><strong>Coordinates:</strong> ${latestResult.coordinates.lat.toFixed(6)}, ${latestResult.coordinates.lon.toFixed(6)}</p>`;
            }
            
            // Display address if available
            if (latestResult.address) {
                captionHTML += `<p><strong>Address:</strong> ${latestResult.address}</p>`;
            }
            
            // Display OCR results if available
            if (latestResult.ocr_result) {
                captionHTML += '<p><strong>OCR Results:</strong></p><ul>';
                if (latestResult.ocr_result.final) {
                    captionHTML += `<li>Final: ${latestResult.ocr_result.final}</li>`;
                }
                if (latestResult.ocr_result.norm) {
                    captionHTML += `<li>Normalized: ${latestResult.ocr_result.norm}</li>`;
                }
                if (latestResult.ocr_result.joined) {
                    captionHTML += `<li>Joined: ${latestResult.ocr_result.joined}</li>`;
                }
                if (latestResult.ocr_result.confidence) {
                    captionHTML += `<li>Confidence: ${(latestResult.ocr_result.confidence * 100).toFixed(2)}%</li>`;
                }
                captionHTML += '</ul>';
            }
            
            // Display buildings detection if available
            if (latestResult.buildings && latestResult.buildings.length > 0) {
                captionHTML += '<p><strong>Buildings Detected:</strong></p><ul>';
                latestResult.buildings.forEach((building, index) => {
                    captionHTML += `<li>Building ${index + 1}: Confidence ${building.confidence.toFixed(2)}`;
                    if (building.area) {
                        captionHTML += `, Area: ${building.area}`;
                    }
                    captionHTML += '</li>';
                });
                captionHTML += '</ul>';
            }
            
            // Display processed at timestamp
            if (latestResult.processed_at) {
                captionHTML += `<p><strong>Processed at:</strong> ${new Date(latestResult.processed_at).toLocaleString()}</p>`;
            }
            
            captionHTML += '</div>';
        } else {
            captionHTML += '<p class="processing-pending">Processing pending...</p>';
        }
        
        modalCaption.innerHTML = captionHTML;
        modal.style.display = 'block';
    }
    
    // Load photos when page loads
    loadPhotos();
});
