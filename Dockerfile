# Use the official Python image as the base image
FROM python:3.10-slim-bullseye

# Create a non-root user
RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser

# Set up a virtual environment
ENV VIRTUAL_ENV=/home/appuser/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the requirements file and install dependencies
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --chown=appuser:appuser . .

# Expose the port the app will run on
EXPOSE 8050

# Start the application
CMD ["gunicorn", "app:server", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8050"]
