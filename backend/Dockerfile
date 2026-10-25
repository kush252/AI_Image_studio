FROM python:3.11-slim

# OpenCV needs these even in headless builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
EXPOSE 8000

# ONE worker. The image store and render cache are module-level singletons
# holding numpy arrays in process memory, so a second worker is a second store:
# an upload handled by one process is invisible to the other. Measured with
# --workers 2, the same image id returned: 200 404 404 404 200 200 404 200.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
