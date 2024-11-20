# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the application code to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the required port (default: 8080 for Render)
EXPOSE 8080

# Command to start the Gunicorn server
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:8080", "app:app"]
