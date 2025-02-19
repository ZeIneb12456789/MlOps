# In flask_interface/Dockerfile
FROM python:3.8-slim

# Set work directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the necessary port
EXPOSE 8000

# Start the app
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]

