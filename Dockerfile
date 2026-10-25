# Single-container deployment: one image serving both the API and the UI.
#
# Build from the PROJECT ROOT, not from backend/, because the frontend is
# compiled here rather than on the developer's machine:
#
#     docker build -t ipcv-platform .
#     docker run -p 8000:8000 ipcv-platform
#
# Building the UI inside the image is deliberate. The older backend/Dockerfile
# picked up whatever happened to be sitting in app/static on the host, so the
# image silently shipped a stale UI - or no UI at all - if you forgot to run
# `npm run build` first.

# ---------------------------------------------------------------- UI build --
FROM node:20-slim AS web
WORKDIR /web

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# vite.config.ts writes to ../backend/app/static, which is outside this stage's
# working copy, so the output is redirected to a path we can copy from.
RUN npm run build -- --outDir /web/dist --emptyOutDir

# ------------------------------------------------------------------- runtime --
FROM python:3.11-slim

# OpenCV needs these even in headless builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
# Replace any statics copied from the build context with the ones just built.
RUN rm -rf ./app/static
COPY --from=web /web/dist ./app/static

EXPOSE 8000

# ONE worker, deliberately. The image store and the render cache are
# module-level singletons holding numpy arrays in process memory, so a second
# worker is a second store: an upload handled by one process is invisible to
# the other, and roughly half of all follow-up requests 404. Measured, with
# --workers 2: 200 404 404 404 200 200 404 200.
#
# Little throughput is lost. OpenCV and NumPy already parallelise across cores
# inside a single process, and the app caps their thread pools so one request
# cannot starve another. To scale beyond one core, run more CONTAINERS behind a
# load balancer with session affinity - or move the store to Redis first.
#
# Shell form, so $PORT expands: Cloud Run, Render and Railway inject the port
# they want the container to listen on, and a hardcoded 8000 fails their health
# check. Falls back to 8000 for local runs.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
