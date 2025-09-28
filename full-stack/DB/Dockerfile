# syntax=docker/dockerfile:1

# Use build arg to pick Postgres version; defaults to lightweight alpine variant
ARG POSTGRES_VERSION=16-alpine
FROM postgres:${POSTGRES_VERSION}

# Configure default credentials and database (override at runtime via env vars)
ENV POSTGRES_USER=appuser \
    POSTGRES_PASSWORD=apppassword \
    POSTGRES_DB=appdb

# Expose PostgreSQL default port
EXPOSE 5432

# Persist database files
VOLUME ["/var/lib/postgresql/data"]

# Basic healthcheck to ensure DB is ready to accept connections
HEALTHCHECK --interval=10s --timeout=5s --retries=5 CMD pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" || exit 1


