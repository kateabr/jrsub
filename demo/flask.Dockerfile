FROM python:3.9-slim-buster
WORKDIR /app

RUN pip install jsub flask gunicorn

COPY flask.py flask.py

ENV PORT=5000

CMD gunicorn -b :$PORT flask:app
