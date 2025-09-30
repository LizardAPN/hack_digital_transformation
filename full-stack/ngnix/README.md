# Nginx Static File Server and Web Interface

This project contains a Docker container configuration for serving static HTML, CSS, and JavaScript files using nginx. It provides the web interface for the building detection application.

## Files Included

- `login.html` - Login page
- `register.html` - Registration page
- `workbench.html` - Main application interface
- `styles.css` - Stylesheet for all pages
- `login.js` - JavaScript for login functionality
- `register.js` - JavaScript for registration functionality
- `workbench.js` - JavaScript for main application functionality
- `logout.js` - JavaScript for logout functionality

## Docker Setup

### Build the Docker Image

```bash
docker build -t nginx-static-server .
```

### Run the Container

```bash
docker run -d -p 8080:80 --name static-server nginx-static-server
```

### Access the Pages

After running the container, you can access the pages at:
- Login page: http://localhost:8080/login.html
- Registration page: http://localhost:8080/register.html
- Workbench (main application): http://localhost:8080/workbench.html

### Stop the Container

```bash
docker stop static-server
```

### Remove the Container

```bash
docker rm static-server
```

## Web Interface Features

The web interface provides the following functionality:

### Authentication
- User registration and login
- Session management
- Logout functionality

### Photo Management
- Upload photos for processing
- View uploaded photos in a grid layout
- View photo details in a modal window

### Building Detection
- Automatic coordinate and address detection for uploaded photos
- Display of detected buildings with bounding boxes
- OCR results display

### Search Functionality
- Search for photos by coordinates (latitude, longitude)
- Search for photos by address
- Display of search results with distance information

### Data Export
- Export processing results to XLSX format
- Download exported data

## Configuration

The nginx server is configured to:
- Serve static files from the `/usr/share/nginx/html/` directory
- Listen on port 80
- Include security headers
- Enable gzip compression for text-based files
- Proxy API requests to backend services
