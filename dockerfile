FROM python:3.10-slim

# System libs for OpenCV/ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Ensure subprocess uses in-container Python
ENV PYTHON_PATH=python
EXPOSE 8080

# You can also use gunicorn if you prefer
CMD ["python", "app_clean.py"]