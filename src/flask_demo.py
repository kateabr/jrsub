import time

from flask import Flask, render_template, request

from src.warodai import WarodaiLoader
from src.yarxi import YarxiLoader

app = Flask(__name__)
yd = YarxiLoader().load()
wd = WarodaiLoader().load()


@app.route('/')
def hello_world():
    dictionary = request.args.get('dictionary')
    lexeme = request.args.get('lexeme')
    reading = request.args.get('reading')
    result = []
    start = time.time()
    if dictionary == 'y':
        result = yd.lookup(lexeme=lexeme, reading=reading)
    elif dictionary == 'w':
        result = wd.lookup(lexeme=lexeme, reading=reading)
    end = time.time()
    return render_template('index.html', result=result, dictionary=dictionary, lexeme=lexeme, reading=reading,
                           exec_time=round(end - start, 2))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
