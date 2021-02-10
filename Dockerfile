FROM python:3.9-alpine

WORKDIR /app

RUN pip install flask jtran tqdm

COPY src src
COPY dictionaries/yarxi.jtdb dictionaries/yarxi.jtdb
COPY dictionaries/warodai.jtdb dictionaries/warodai.jtdb

ENV PORT=5000
ENV FLASK_RUN_PORT=$PORT

EXPOSE $PORT

ENTRYPOINT PYTHONPATH=$PYTHONPATH:/app python src/flask_demo.py