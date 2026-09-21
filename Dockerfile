FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e . \
    && python scripts/train_model.py \
    && python scripts/generate_mlops_artifacts.py \
    && python scripts/generate_evidently_report.py

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
