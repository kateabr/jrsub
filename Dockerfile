FROM python:3.9-slim-buster

WORKDIR /app

RUN pip install flask jtran tqdm gunicorn

COPY src src
COPY dictionaries/yarxi.jtdb dictionaries/yarxi.jtdb
COPY dictionaries/warodai.jtdb dictionaries/warodai.jtdb

ENV PORT=5000

CMD PYTHONPATH=$PYTHONPATH:/app gunicorn --log-level debug --chdir src -b :$PORT flask_demo:app
