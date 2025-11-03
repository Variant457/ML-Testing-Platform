FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY templates/ ./templates/
COPY static/ ./static/
COPY app.py ./
COPY config.py ./
COPY ml_models.py ./
COPY models.py ./
COPY utils.py ./
COPY .env ./

VOLUME /app/instance
VOLUME /app/uploads

EXPOSE 5000

CMD ["python", "app.py"]