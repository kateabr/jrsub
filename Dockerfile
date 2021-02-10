FROM python:3.9-alpine

WORKDIR /app

RUN pip install flask jtran tqdm gunicorn

COPY src src
COPY dictionaries/yarxi.jtdb dictionaries/yarxi.jtdb
COPY dictionaries/warodai.jtdb dictionaries/warodai.jtdb

ENV PORT=5000

ENTRYPOINT PYTHONPATH=$PYTHONPATH:/app gunicorn --chdir src flask_demo:app -b 0.0.0.0:$PORT
