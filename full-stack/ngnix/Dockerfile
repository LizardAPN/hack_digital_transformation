# Use the official nginx image as the base image
FROM nginx:alpine

# Copy the static files to the nginx html directory
COPY *.html /usr/share/nginx/html/
COPY *.css /usr/share/nginx/html/
COPY *.js /usr/share/nginx/html/

# Copy the nginx configuration file
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port 80
EXPOSE 80

# Start nginx when the container runs
CMD ["nginx", "-g", "daemon off;"]