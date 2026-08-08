FROM python:3.10-slim

WORKDIR /app

COPY backend/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.13.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    flask==3.1.3 \
    flask-cors==6.0.1 \
    numpy==2.2.6 \
    scipy==1.15.3 \
    scikit-learn==1.5.2 \
    joblib==1.5.3 \
    threadpoolctl==3.6.0

COPY backend ./backend
COPY frontend ./frontend
COPY data ./data
COPY saved_models ./saved_models

WORKDIR /app/backend

EXPOSE 5000

CMD ["python", "-u", "app.py"]