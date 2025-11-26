FROM python:3.10

WORKDIR /code

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["gunicorn", "app:app", "--worker-class", "eventlet", "--workers", "1", "--bind", "0.0.0.0:7860"]
