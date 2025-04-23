FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for output
RUN mkdir -p analysis_results monitoring_results

# Set default command
ENTRYPOINT ["python", "atc_monitor.py"]

# Default duration (can be overridden)
CMD ["30"]