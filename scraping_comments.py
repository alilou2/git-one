import numpy as np
import facebook_scraper
from facebook_scraper import enable_logging
from classification_utils import predictlistComments
from model_loader import models , tokenizerbrt
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid
from collections import OrderedDict
import datetime
from bson.objectid import ObjectId


enable_logging()

client = MongoClient("mongodb://localhost:27017")
db = client['mydatabase']


post = {
    'post_id': {
        'type': int,
        'required': True,
    },
    'post_text': {
        'type': str,
        'required': True,
    },
    'post_image': {
        'type': str,
        'required': True,
    },
    'num_com': {
        'type': int,
        'required': True,
    },
    'post_likes': {
        'type': int,
        'required': True,
    },
    'post_time': {
        'type': datetime.datetime,
        'required': True,
    },
    'comments': [
        {
            'comment_id': {
                'type': int,
                'required': True,
            },
            'comment_text': {
                'type': str,
                'required': True,
            },
            'commenter_name': {
                'type': str,
            },
            'predicted_class':{
                'type': str,
                'required': True,
            },
        },
    ],
    'class_distribution':{
        'type': 'object',
        'required': True
    }
}

def calculate_class_distribution(classes):
    total = len(classes)
    distribution = {}
    for cls in classes:
        if cls in distribution:
            distribution[cls] += 1
        else:
            distribution[cls] = 1

    percentage_distribution = {}
    for cls, count in distribution.items():
        percentage = (count / total) * 100
        percentage_distribution[cls] = np.round(percentage, 2)

    return percentage_distribution

def scraping(num_post):
    #cookies = "C:/Users/BIGNETWORK/Desktop/PFE/dataset/cookies.txt" #it is necessary to generate a Facebook cookie file
    posts_collection = db["posts_collection"]

    page_id = '182557711758412'

    posts = []
    for post in facebook_scraper.get_posts(page_id,pages=num_post, options={"comments": "generator", "HQ_images": True}):
        posts.append(post)
         # Create a new document based on the schema
    
    # Iterate over the retrieved posts
    for post in posts:
        post_id = post['post_id']
        print(post_id)
        # Add your desired code logic here to process each post as needed
        # For example, extract relevant information, comments, etc.
        new_post = {
            'post_id': post_id,
            'post_text': post["post_text"] if 'post_text' in post else None,
            'post_image': post['image'],
            'num_com': post['comments_full_count'] if 'comments_full_count' in post else None,
            'post_likes': post['likes'] if 'likes' in post else None,
            'post_time': post['time'],
            'comments': [],
        }

        # Retrieve the comments for the current post
        comments = post["comments_full"]
        for comment in comments:
            comment_text = comment['comment_text']
            predicted_class = predictlistComments(comment_text, models, tokenizerbrt)
            new_comment = {
                  'comment_id': comment['comment_id'],
                  'comment_text': comment_text,
                  'commenter_name': comment['commenter_name'],
                  'predicted_class': predicted_class,
                }
            new_post['comments'].append(new_comment)
        # Insert the new post document into the MongoDB collection
        predicted_classes = [comment['predicted_class'] for comment in new_post['comments']]
        class_distribution = calculate_class_distribution(predicted_classes)
        new_post['class_distribution'] = class_distribution

        # Insert the new post document into the MongoDB collection
        posts_collection.insert_one(new_post)
    return posts_collection

#posts_collection = scraping(4)


def fetch_latest_posts(n):
    posts_collection = db["posts_collection"]
    # Fetch the latest 'n' posts from the collection
    posts = list(posts_collection.find().sort('post_time', -1).limit(n))
    print(posts)
    return posts

