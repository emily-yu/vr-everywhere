from flask import Flask
from flask import request

app = Flask(__name__)

@app.route("/")
def hello():
    print ("yahallo")
    return "yahallo"

@app.route('/check')
def check():
    action = request.args.get("action")

    if(action == "start"):
        text_file = open('blinds.txt', 'w')
        text_file.write(action)
    elif (action == "stop"):
        text_file = open('blinds.txt', 'w')
        text_file.write('')
    return

if __name__ == "__main__":
    app.run()
