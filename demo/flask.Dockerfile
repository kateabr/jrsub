FROM python:3.9-slim-buster
WORKDIR /app

RUN pip install jrsub flask gunicorn

COPY flask.py app.py

ENV PORT=5000

CMD gunicorn -b :$PORT app:app
