from flask import Flask, render_template, request

from src.warodai import WarodaiLoader
from src.yarxi import YarxiLoader

app = Flask(__name__)
yd = YarxiLoader().load('./dictionaries/yarxi.jtdb')
wd = WarodaiLoader().load('./dictionaries/warodai.jtdb')


@app.route('/translate')
def translate():
    dictionary = request.args.get('dictionary')
    lexeme = request.args.get('lexeme')
    reading = request.args.get('reading')
    if dictionary == 'y':
        return render_template('index.html', result=yd.lookup(lexeme=lexeme, reading=reading), dictionary=dictionary)
    elif dictionary == 'w':
        return render_template('index.html', result=wd.lookup(lexeme=lexeme, reading=reading), dictionary=dictionary)
    return render_template('index.html', result=[])


@app.route('/')
def hello_world():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
