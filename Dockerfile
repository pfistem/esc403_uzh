FROM python:3.10.6-slim

# Create a non-root user
RUN useradd --create-home appuser
USER appuser

WORKDIR /app

# Set up a virtual environment
ENV VIRTUAL_ENV=/home/appuser/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8050

CMD ["gunicorn", "app:server", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8050"]
