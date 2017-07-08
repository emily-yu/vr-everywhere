from flask import Flask, request
import urllib.request
# from bottle import route, run, template, static_file, get, post, request
import requests
# from secret import getSID, getAuth, getAuthy
import string
import re
import json

app = Flask(__name__)

@app.route("/")
def hello():
    return "hey its me"

@app.route("/send")
def send():
	image = request.args.get("input")
	print(image)
	print(image.split('%', (image.count('%')))) # should be each of the images in base64
	return "asdf"

if __name__ == '__main__':
        app.run()

