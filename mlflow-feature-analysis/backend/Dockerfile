FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create uploads directory
RUN mkdir -p ./uploads

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "main.py"]

