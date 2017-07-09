from flask import Flask, request
import urllib.request
# from bottle import route, run, template, static_file, get, post, request
import requests
# from secret import getSID, getAuth, getAuthy
import string
import re
import json
import base64

app = Flask(__name__)

images = []

@app.route("/")
def hello():
    return "hey its me"

@app.route("/send")
def send():
	# https://4adacd3b.ngrok.io/send?input=blablablhalblablhabhlalh
	# this part is a little boosted but nrlly
	image = request.args.get("input")
	print ("YAALLO MAREKR")
	print(image)
	decoded = base64.urlsafe_b64decode(image)
	decoded = decoded.replace('.', '=')
	print(decoded)
	images.append(decoded)
	return decoded

if __name__ == '__main__':
        app.run()

