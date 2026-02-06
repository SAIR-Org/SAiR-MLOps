FROM python:3.11-slim 
WORKDIR /app 
COPY app.py iris_model.pkl requirements.txt /app/
RUN pip install -r requirements.txt 
EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]