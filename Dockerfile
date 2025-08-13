FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY artifacts/model.joblib artifacts/feature_columns.json artifacts/
COPY inference/ inference/

COPY entrypoint.sh .
ENTRYPOINT ["/app/entrypoint.sh"]

EXPOSE 8080
CMD ["python","-m","uvicorn","inference.predict:app","--host","0.0.0.0","--port","8080"]