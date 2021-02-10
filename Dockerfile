FROM python:3.9-slim-buster

WORKDIR /app

RUN pip install flask jtran tqdm gunicorn

COPY src src
COPY dictionaries/yarxi.jtdb dictionaries/yarxi.jtdb
COPY dictionaries/warodai.jtdb dictionaries/warodai.jtdb

ENV PORT=5000

ENTRYPOINT ["/bin/bash", "-c", "PYTHONPATH=$PYTHONPATH:/app gunicorn --log-level debug --chdir src flask_demo:app -b 0.0.0.0:$PORT"]
