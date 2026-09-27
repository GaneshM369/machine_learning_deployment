# Use lightweight official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirement definition first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all repository files (app.py, model.pkl, scaler.joblib, etc.)
COPY . .

# Expose default port
EXPOSE 5000

# Start app with Gunicorn (supports dynamic PORT provided by host platforms)
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-5000} app:app"]
