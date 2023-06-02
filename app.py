from flask import Flask, render_template, request, jsonify, make_response
#from model_loader import load_models
from classification_utils import predictComments
from model_loader import models, tokenizerbrt
from flask_cors import CORS
from pymongo import MongoClient
from scraping_comments import fetch_latest_posts
import json
from bson import json_util

def parse_json(data):
    return json.loads(json_util.dumps(data))
fetch_latest_posts

client = MongoClient("mongodb://localhost:27017")
db = client['mydatabase']

app = Flask(__name__)

#models ,tokenizerbrt = load_models()
CORS(app)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.json.get('text')
    if text is None:
        return jsonify({'error': 'Text is missing'})

    # Make prediction using the processed text
    response = predictComments(text, models, tokenizerbrt)
    
    response = {k: v.item() for k, v in response.items()}
    return jsonify(response)

@app.route('/latest-posts', methods=['GET'])
def get_latest_posts():
    # Get the 'n' parameter from the request query string
    n = int(request.args.get('n', 4))  # default to 10 if 'n' is not provided
    # Fetch the latest 'n' posts
    posts = fetch_latest_posts(n)
    postsAsjson = json.dumps(posts, default=str);
    response =  make_response(postsAsjson)
    # Prepare the response as a list of formatted documents
    # response = [format_document(post) for post in posts]
    response.headers['Content-Type'] = 'application/json'
    return response
'''# allow both GET and POST requests
@app.route('/analytics', methods=['GET', 'POST'])
def'''

if __name__ == '__main__':
    app.run(debug=False)
