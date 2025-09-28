# Nginx Static File Server

This project contains a Docker container configuration for serving static HTML and CSS files using nginx.

## Files Included

- `login.html` - Login page
- `register.html` - Registration page
- `styles.css` - Stylesheet for both pages

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

### Stop the Container

```bash
docker stop static-server
```

### Remove the Container

```bash
docker rm static-server
```

## Configuration

The nginx server is configured to:
- Serve static files from the `/usr/share/nginx/html/` directory
- Listen on port 80
- Include security headers
- Enable gzip compression for text-based files