// Workbench JavaScript functionality

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const uploadButton = document.getElementById('upload-button');
    const photoUploadInput = document.getElementById('photo-upload');
    const photoGrid = document.getElementById('photo-grid');
    const uploadStatus = document.getElementById('upload-status');
    const addPhotoCard = document.getElementById('add-photo-card');
    
    // Search elementsf
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const searchResults = document.getElementById('search-results');
    
    // Export elements
    const exportXlsxButton = document.getElementById('export-xlsx-button');
    
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
    
    // Event listener for search button
    searchButton.addEventListener('click', function() {
        const query = searchInput.value.trim();
        if (query) {
            searchPhotos(query);
        }
    });
    
    // Event listener for export XLSX button
    exportXlsxButton.addEventListener('click', function() {
        exportToXlsx();
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
            // Для каждого фото запрашиваем результаты обработки
            const photosWithResults = data.photos.map(photo => {
                return fetch(`/api/results/photo/${photo.id}`, {
                    method: 'GET',
                    credentials: 'include'
                })
                .then(response => {
                    if (response.ok) {
                        return response.json();
                    } else {
                        return null;
                    }
                })
                .then(result => {
                    return {...photo, processing_result: result};
                })
                .catch(error => {
                    console.error('Error loading processing result for photo:', photo.id, error);
                    return {...photo, processing_result: null};
                });
            });
            
            // Ждем завершения всех запросов
            Promise.all(photosWithResults)
                .then(photos => {
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
            
            // Determine processing status
            let statusHtml = '';
            if (photo.processing_result) {
                if (photo.processing_result.error) {
                    statusHtml = `<p class="photo-status error">Ошибка обработки</p>`;
                } else if (photo.processing_result.coordinates) {
                    const coords = photo.processing_result.coordinates;
                    statusHtml = `<p class="photo-status success">Координаты: ${coords.lat?.toFixed(6)}, ${coords.lon?.toFixed(6)}</p>`;
                } else {
                    statusHtml = `<p class="photo-status processing">Обработка...</p>`;
                }
            } else {
                statusHtml = `<p class="photo-status unknown">Статус неизвестен</p>`;
            }
            
            photoCard.innerHTML = `
                <img src="${photo.photo_url}" alt="${photo.created_at}" class="photo-thumbnail">
                <p class="photo-name">${new Date(photo.created_at).toLocaleString()}</p>
                ${statusHtml}
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
        
        // Build caption with processing results
        let captionHtml = `<h3>${new Date(photo.created_at).toLocaleString()}</h3>`;
        captionHtml += `<p>Uploaded: ${new Date(photo.created_at).toLocaleString()}</p>`;
        
        if (photo.processing_result) {
            if (photo.processing_result.error) {
                captionHtml += `<p class="modal-error">Ошибка обработки: ${photo.processing_result.error}</p>`;
            } else {
                if (photo.processing_result.coordinates) {
                    const coords = photo.processing_result.coordinates;
                    captionHtml += `<p class="modal-coordinates"><strong>Координаты:</strong> ${coords.lat?.toFixed(6)}, ${coords.lon?.toFixed(6)}</p>`;
                }
                
                if (photo.processing_result.address) {
                    captionHtml += `<p class="modal-address"><strong>Адрес:</strong> ${photo.processing_result.address}</p>`;
                }
                
                if (photo.processing_result.processed_at) {
                    captionHtml += `<p class="modal-processed"><strong>Обработано:</strong> ${new Date(photo.processing_result.processed_at).toLocaleString()}</p>`;
                }
            }
        } else {
            captionHtml += `<p class="modal-info">Результаты обработки еще не доступны</p>`;
        }
        
        modalCaption.innerHTML = captionHtml;
        modal.style.display = 'block';
    }
    
    // Function to search photos by coordinates or address
    function searchPhotos(query) {
        searchResults.innerHTML = '<p>Searching...</p>';
        
        // Check if query is coordinates (lat,lon format)
        const coordMatch = query.match(/^(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)$/);
        
        if (coordMatch) {
            // Search by coordinates
            const lat = parseFloat(coordMatch[1]);
            const lon = parseFloat(coordMatch[2]);
            
            fetch('/api/search/by_coordinates', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ lat: lat, lon: lon, radius_km: 1.0 }),
                credentials: 'include'
            })
            .then(response => response.json())
            .then(data => {
                displaySearchResults(data);
            })
            .catch(error => {
                searchResults.innerHTML = '<p>Error searching by coordinates.</p>';
                console.error('Search error:', error);
            });
        } else {
            // Search by address
            fetch('/api/search/by_address', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ address: query }),
                credentials: 'include'
            })
            .then(response => response.json())
            .then(data => {
                displaySearchResults(data);
            })
            .catch(error => {
                searchResults.innerHTML = '<p>Error searching by address.</p>';
                console.error('Search error:', error);
            });
        }
    }
    
    // Function to display search results
    function displaySearchResults(results) {
        if (results.length === 0) {
            searchResults.innerHTML = '<p>No results found.</p>';
            return;
        }
        
        let html = '<h4>Search Results:</h4><div class="search-results-grid">';
        results.forEach(result => {
            html += `
                <div class="search-result-item">
                    <p><strong>Coordinates:</strong> ${result.coordinates.lat}, ${result.coordinates.lon}</p>
                    <p><strong>Address:</strong> ${result.address || 'N/A'}</p>
                    <p><strong>Distance:</strong> ${result.distance_km ? result.distance_km.toFixed(2) + ' km' : 'N/A'}</p>
                    <p><strong>Processed:</strong> ${new Date(result.processed_at).toLocaleString()}</p>
                </div>
            `;
        });
        html += '</div>';
        
        searchResults.innerHTML = html;
    }
    
    // Function to export data to XLSX
    function exportToXlsx() {
        window.location.href = '/export/results/xlsx';
    }
    
    // Load photos when page loads
    loadPhotos();
    
    // Function to check for photo processing results every second
    function checkPhotoStatus() {
        // Find all photo cards with unknown status
        const unknownStatusPhotos = document.querySelectorAll('.photo-status.processing');
        
        unknownStatusPhotos.forEach(photoStatusElement => {
            // Get the parent photo card
            const photoCard = photoStatusElement.closest('.photo-card');
            if (!photoCard) return;
            
            // Get the photo URL from the image element
            const imgElement = photoCard.querySelector('.photo-thumbnail');
            if (!imgElement) return;
            
            const photoUrl = imgElement.src;
            if (!photoUrl) return;
            
            // Extract photo ID from URL (assuming format: .../photo_id.jpg)
            const urlParts = photoUrl.split('/');
            const photoId = urlParts[urlParts.length - 1];
            
            // Send request to check photo processing status
            fetch(`http://localhost/results/photo/${photoId}`)
                .then(response => {
                    if (response.ok) {
                        return response.json();
                    } else {
                        throw new Error('Failed to fetch photo status');
                    }
                })
                .then(data => {
                    // Check if we have coordinates in the response
                    if (data && data[4] && data[4].lat !== undefined && data[4].lon !== undefined) {
                        const coords = data[4];
                        // Update the status element with coordinates
                        photoStatusElement.textContent = `Координаты: ${coords.lat.toFixed(6)}, ${coords.lon.toFixed(6)}`;
                        photoStatusElement.className = 'photo-status success';
                    }
                })
                .catch(error => {
                    console.error('Error checking photo status:', error);
                });
        });
    }
    
    // Run the checkPhotoStatus function every second
    setInterval(checkPhotoStatus, 1000);
});
