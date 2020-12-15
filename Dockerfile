# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.7-slim-buster

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
ADD requirements.txt .
RUN python -m pip install -r requirements.txt

# Set directory
WORKDIR /app
ADD . /app

# Switching to a non-root user
RUN useradd appuser && chown -R appuser /app
USER appuser

# Set gunicorn as webserver
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "mysite.mysite.wsgi"]
