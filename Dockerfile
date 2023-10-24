
FROM python:3.9


# Copy files to the container

COPY requirements.txt /app/

# Set working directory to previously added app directory
WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install python-multipart

# Copy files
COPY config config/
COPY utils ./utils/
COPY deploy.py .
COPY detect_images.py .
COPY detect_video.py .
COPY ssd/ ssd/

RUN mkdir -p weight
RUN mkdir -p data

EXPOSE 8000
CMD ["uvicorn", "deploy:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]