// Workbench JavaScript functionality

// Global variable to store current workspace
let currentWorkspace = null;
let workspaces = [];

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const uploadButton = document.getElementById('upload-button');
    const zipButton = document.getElementById('zip-button');
    const photoUploadInput = document.getElementById('photo-upload');
    const zipUploadInput = document.getElementById('zip-upload');
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

    zipButton.addEventListener('click', function() {
        zipUploadInput.click();
    });
    
    // Event listener for file selection
    photoUploadInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            uploadPhoto(file);
        }
    });

    // Event listener for file selection
    zipUploadInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            uploadZip(file);
            //123
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
        uploadStatus.textContent = 'Загрузка...';
        uploadStatus.className = 'upload-status uploading';
        
        const formData = new FormData();
        formData.append('file', file);
        
        // Добавляем workspace_id если выбрана рабочая область
        let url = '/api/photo_upload';
        if (currentWorkspace) {
            url += `?workspace_id=${currentWorkspace.id}`;
        }
        
        fetch(url, {
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
            uploadStatus.textContent = 'Загрузка успешна!';
            uploadStatus.className = 'upload-status success';
            return loadPhotos(1, 10); // Refresh photo grid
        })
        .then(() => {
            photoUploadInput.value = ''; // Reset file input to allow re-upload of same file
            setTimeout(() => {
                uploadStatus.textContent = '';
                uploadStatus.className = 'upload-status';
            }, 3000);
        })
        .catch(error => {
            uploadStatus.textContent = 'Загрузка не удалась. Пожалуйста, попробуйте снова.';
            uploadStatus.className = 'upload-status error';
            console.error('Ошибка загрузки:', error);
            photoUploadInput.value = ''; // Reset file input to allow re-upload of same file
        });
    }

        // Function to upload photo
    function uploadZip(file) {
        uploadStatus.textContent = 'Загрузка...';
        uploadStatus.className = 'upload-status uploading';
        
        const formData = new FormData();
        formData.append('file', file);
        
        // Добавляем workspace_id если выбрана рабочая область
        let url = '/api/zip_upload';
        if (currentWorkspace) {
            url += `?workspace_id=${currentWorkspace.id}`;
        }
        
        fetch(url, {
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
            uploadStatus.textContent = 'Загрузка успешна!';
            uploadStatus.className = 'upload-status success';
            return loadPhotos(1, 10); // Refresh photo grid
        })
        .then(() => {
            zipUploadInput.value = ''; // Reset file input to allow re-upload of same file
            setTimeout(() => {
                uploadStatus.textContent = '';
                uploadStatus.className = 'upload-status';
            }, 3000);
        })
        .catch(error => {
            uploadStatus.textContent = 'Загрузка не удалась. Пожалуйста, попробуйте снова.';
            uploadStatus.className = 'upload-status error';
            console.error('Ошибка загрузки:', error);
            zipUploadInput.value = ''; // Reset file input to allow re-upload of same file
        });
    }
    
    // Function to load photos with pagination
    function loadPhotos(page = 1, limit = 10) {
        const photoGrid = document.getElementById('photo-grid');
        
        // Show loading indicator
        const loadingElement = document.createElement('div');
        loadingElement.className = 'loading';
        loadingElement.textContent = 'Загрузка фотографий...';
        loadingElement.id = 'photos-loading';
        
        // Clear existing photos except the add photo card and pagination
        const addPhotoCard = document.getElementById('add-photo-card');
        const paginationContainer = document.getElementById('pagination-container');
        
        // Store references and remove pagination temporarily
        const nextSibling = paginationContainer ? paginationContainer.nextSibling : null;
        if (paginationContainer) {
            paginationContainer.remove();
        }
        
        // Clear photo grid but keep add photo card
        photoGrid.innerHTML = '';
        if (addPhotoCard) {
            photoGrid.appendChild(addPhotoCard);
        }
        photoGrid.appendChild(loadingElement);
        
        // Добавляем workspace_id если выбрана рабочая область
        let url = `/api/photos?page=${page}&limit=${limit}`;
        if (currentWorkspace) {
            url += `&workspace_id=${currentWorkspace.id}`;
        }
        
        return fetch(url, {
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
                    console.error('Ошибка загрузки результатов обработки для фото:', photo.id, error);
                    return {...photo, processing_result: null};
                });
            });
            
            // Ждем завершения всех запросов
            return Promise.all(photosWithResults);
        })
        .then(photos => {
            // Remove loading indicator
            const loadingElement = document.getElementById('photos-loading');
            if (loadingElement) {
                loadingElement.remove();
            }
            
            // Re-add pagination container if it existed
            if (paginationContainer) {
                photoGrid.parentNode.insertBefore(paginationContainer, photoGrid.nextSibling);
            }
            
            displayPhotos(photos, page, limit);
        })
        .catch(error => {
            console.error('Error loading photos:', error);
            // Remove loading indicator
            const loadingElement = document.getElementById('photos-loading');
            if (loadingElement) {
                loadingElement.remove();
            }
            
            // Re-add pagination container if it existed
            if (paginationContainer) {
                photoGrid.parentNode.insertBefore(paginationContainer, photoGrid.nextSibling);
            }
            
            // Show error message
            const errorElement = document.createElement('div');
            errorElement.className = 'error';
            errorElement.textContent = 'Ошибка загрузки фотографий';
            photoGrid.appendChild(errorElement);
            
            throw error; // Re-throw to be caught by caller
        });
    }
    
    // Function to display photos in grid with pagination
    function displayPhotos(photos, currentPage = 1, limit = 10) {
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
        
        // Add pagination controls after the photo grid
        displayPagination(currentPage, limit, photos.length);
    }
    
    // Function to display pagination controls
    function displayPagination(currentPage, limit, photoCount) {
        // Create pagination container if it doesn't exist
        let paginationContainer = document.getElementById('pagination-container');
        if (!paginationContainer) {
            paginationContainer = document.createElement('div');
            paginationContainer.id = 'pagination-container';
            paginationContainer.className = 'pagination-container';
            photoGrid.parentNode.insertBefore(paginationContainer, photoGrid.nextSibling);
        }
        
        // Clear existing pagination controls
        paginationContainer.innerHTML = '';
        
        // Create pagination controls
        const paginationControls = document.createElement('div');
        paginationControls.className = 'pagination-controls';
        
        // Previous button
        if (currentPage > 1) {
            const prevButton = document.createElement('button');
            prevButton.className = 'pagination-button';
            prevButton.textContent = 'Назад';
            prevButton.addEventListener('click', function() {
                loadPhotos(currentPage - 1, limit);
            });
            paginationControls.appendChild(prevButton);
        }
        
        // Page numbers (show up to 5 pages around current page)
        const startPage = Math.max(1, currentPage - 2);
        const endPage = startPage + 4;
        
        for (let i = startPage; i <= endPage; i++) {
            const pageButton = document.createElement('button');
            pageButton.className = 'pagination-button' + (i === currentPage ? ' active' : '');
            pageButton.textContent = i;
            pageButton.addEventListener('click', function() {
                loadPhotos(i, limit);
            });
            paginationControls.appendChild(pageButton);
        }
        
        // Next button (always show if there might be more photos)
        // We'll show it if we got exactly 10 photos (might be more on next page)
        if (photoCount === limit) {
            const nextButton = document.createElement('button');
            nextButton.className = 'pagination-button';
            nextButton.textContent = 'Вперед';
            nextButton.addEventListener('click', function() {
                loadPhotos(currentPage + 1, limit);
            });
            paginationControls.appendChild(nextButton);
        }
        
        paginationContainer.appendChild(paginationControls);
    }
    
    // Function to open photo in modal
    function openPhotoModal(photo) {
        modalImage.src = photo.photo_url;
        
        // Build caption with processing results
        let captionHtml = `<h3>${new Date(photo.created_at).toLocaleString()}</h3>`;
        captionHtml += `<p>Загружено: ${new Date(photo.created_at).toLocaleString()}</p>`;
        
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
            // Extract photo ID from URL (assuming format: .../photo_id.jpg)
            const urlParts = photo.photo_url.split('/');
            const photoId = urlParts[urlParts.length - 1];
            // Fetch address from API request and take 5th index
            fetch(`/results/photo/${photoId}`)
                .then(response => response.json())
                .then(data => {
                    let addressInfo = 'Результаты обработки еще не доступны';
                    if (data && data.length > 4 && data[5]) {
                        // Taking the 5th index from the response
                        const fifthIndexData = data[5];
                        if (fifthIndexData.address) {
                            addressInfo = fifthIndexData.address;
                        } else if (typeof fifthIndexData === 'string') {
                            addressInfo = fifthIndexData;
                        } else {
                            addressInfo = JSON.stringify(fifthIndexData);
                        }
                    }
                    captionHtml += `<p class="modal-info"><strong>Адрес:</strong> ${addressInfo}</p>`;
                    modalCaption.innerHTML = captionHtml;
                    modal.style.display = 'block';
                })
                .catch(error => {
                    console.error('Ошибка получения адреса:', error);
                    captionHtml += `<p class="modal-info">Результаты обработки еще не доступны</p>`;
                    modalCaption.innerHTML = captionHtml;
                    modal.style.display = 'block';
                });
            return;
        }
        
        modalCaption.innerHTML = captionHtml;
        modal.style.display = 'block';
    }
    
    // Function to search photos by coordinates or address
    function searchPhotos(query) {
        searchResults.innerHTML = '<p>Поиск...</p>';
        
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
                searchResults.innerHTML = '<p>Ошибка поиска по координатам.</p>';
                console.error('Ошибка поиска:', error);
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
                searchResults.innerHTML = '<p>Ошибка поиска по адресу.</p>';
                console.error('Ошибка поиска:', error);
            });
        }
    }
    
    // Function to display search results
    function displaySearchResults(results) {
        if (results.length === 0) {
            searchResults.innerHTML = '<p>Результаты не найдены.</p>';
            return;
        }
        
        let html = '<h4>Результаты поиска:</h4><div class="search-results-grid">';
        results.forEach(result => {
            html += `
                <div class="search-result-item">
                    <p><strong>Координаты:</strong> ${result.coordinates.lat}, ${result.coordinates.lon}</p>
                    <p><strong>Адрес:</strong> ${result.address || 'Н/Д'}</p>
                    <p><strong>Расстояние:</strong> ${result.distance_km ? result.distance_km.toFixed(2) + ' км' : 'Н/Д'}</p>
                    <p><strong>Обработано:</strong> ${new Date(result.processed_at).toLocaleString()}</p>
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
    
    // Load photos when page loads with pagination
    loadPhotos(1, 10);
    
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
            fetch(`/results/photo/${photoId}`)
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
                    console.error('Ошибка проверки статуса фото:', error);
                });
        });
        
        // Find all photo cards with unknown status
        const unknownStatusPhotos_2 = document.querySelectorAll('.photo-status.unknown');
        
        unknownStatusPhotos_2.forEach(photoStatusElement => {
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
            fetch(`/results/photo/${photoId}`)
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
                    console.error('Ошибка проверки статуса фото:', error);
                });
        });
    }
    
    // Run the checkPhotoStatus function every second
    setInterval(checkPhotoStatus, 1000);

    // Workspace functionality
    const sidebarActions = document.querySelector('.sidebar-actions');
    
    // Create workspace button
    const createWorkspaceButton = document.createElement('button');
    createWorkspaceButton.textContent = `+ Создать рабочую область`;
    createWorkspaceButton.className = 'action-button';
    sidebarActions.appendChild(createWorkspaceButton);
    
    // Workspace selector container
    const workspaceSelectorContainer = document.createElement('div');
    workspaceSelectorContainer.className = 'workspace-selector-container';
    workspaceSelectorContainer.style.marginTop = '20px';
    sidebarActions.appendChild(workspaceSelectorContainer);
    
    // Load workspaces and set up event listeners
    loadWorkspaces();
    
    // Event listener for create workspace button
    createWorkspaceButton.addEventListener('click', function() {
        const workspaceName = prompt('Введите название новой рабочей области:');
        if (workspaceName) {
            createWorkspace(workspaceName);
        }
    });
    // Function to load workspaces
function loadWorkspaces() {
    fetch('/api/workspaces', {
        method: 'GET',
        credentials: 'include'
    })
    .then(response => response.json())
    .then(data => {
        workspaces = data.workspaces;
        displayWorkspaceSelector();
    })
    .catch(error => {
        console.error('Ошибка загрузки рабочих областей:', error);
    });
}

// Function to display workspace selector
function displayWorkspaceSelector() {
    const workspaceSelectorContainer = document.querySelector('.workspace-selector-container');
    if (!workspaceSelectorContainer) return;
    
    // Clear existing content
    workspaceSelectorContainer.innerHTML = '';
    
    // Add title
    const title = document.createElement('h3');
    title.textContent = 'Рабочие области';
    title.style.marginTop = '0';
    title.style.marginBottom = '10px';
    workspaceSelectorContainer.appendChild(title);
    
    // Add workspace buttons
    workspaces.forEach(workspace => {
        const workspaceButton = document.createElement('button');
        workspaceButton.textContent = workspace.name;
        workspaceButton.className = 'action-button';
        workspaceButton.style.marginBottom = '8px';
        workspaceButton.style.backgroundColor = currentWorkspace && currentWorkspace.id === workspace.id ? 'var(--primary)' : '';
        
        workspaceButton.addEventListener('click', function() {
            switchWorkspace(workspace);
        });
        
        workspaceSelectorContainer.appendChild(workspaceButton);
    });
}

// Function to create a new workspace
function createWorkspace(name) {
    const requestData = {
        name: name
    };
    
    fetch('/api/workspaces', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
        credentials: 'include'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Reload workspaces
            loadWorkspaces();
            // Switch to the new workspace
            switchWorkspace(data.workspace);
        } else {
            alert('Не удалось создать рабочую область: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Ошибка создания рабочей области:', error);
        alert('Произошла ошибка при создании рабочей области');
    });
}

// Function to switch workspace
function switchWorkspace(workspace) {
    // Show loading indicator
    const photoGrid = document.getElementById('photo-grid');
    const originalContent = photoGrid.innerHTML;
    photoGrid.innerHTML = '<div class="loading">Загрузка рабочей области...</div>';
    
    // Update current workspace
    currentWorkspace = workspace;
    
    // Update UI to show selected workspace
    displayWorkspaceSelector();

    console.log("Loading workspace");

    // Reload photos for the new workspace
    loadPhotos(1, 10).then(() => {
        // Hide loading indicator after photos are loaded
        // The loading message will be automatically replaced by loadPhotos
    }).catch(error => {
        console.error('Ошибка загрузки фотографий:', error);
        photoGrid.innerHTML = originalContent; // Restore original content on error
        alert('Не удалось загрузить фотографии для выбранной рабочей области');
    });
    
    console.log("After Loading instructions");

    // Update export button to use current workspace
    const exportXlsxButton = document.getElementById('export-xlsx-button');
    if (exportXlsxButton) {
        exportXlsxButton.onclick = function() {
            exportToXlsx(workspace.id);
        };
    }
    
    // Clear search results when switching workspace
    const searchResults = document.getElementById('search-results');
    if (searchResults) {
        searchResults.innerHTML = '';
    }
    
    // Reset upload statuses
    const uploadStatus = document.getElementById('upload-status');
    const zipStatus = document.getElementById('zip-status');
    if (uploadStatus) uploadStatus.textContent = '';
    if (zipStatus) zipStatus.textContent = '';
}

// Modified exportToXlsx function to use workspace
function exportToXlsx(workspaceId = null) {
    let url = '/export/results/xlsx';
    if (workspaceId) {
        url += `?workspace_id=${workspaceId}`;
    }
    window.location.href = url;
}


});

