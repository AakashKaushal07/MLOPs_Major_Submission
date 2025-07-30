FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY trained_models/ ./trained_models/

CMD ["python","./src/predict.py"]