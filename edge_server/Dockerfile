FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy all files and folders from current directory to /app in container
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Conditionally create the uploads directory
RUN mkdir -p /app/uploads

# Expose the port the app runs on
EXPOSE 5002

# Create a script to run both server.py and power_logger.py
ENTRYPOINT [ "python", "-u", "/app/server.py"]