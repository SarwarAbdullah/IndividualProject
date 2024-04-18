import os
#import PySimpleGUI as sg
import csv
import pandas as pd
# tweepy is a python library that allows you to interact with the Twitter API
import tweepy

api_key = "ZiK4MQyW3LTXEXNxn3vJEVtFL"
api_key_secret = "r3kjaseLIQQGzJvQmuMBdJzZlvcrQetlcqmyMrrttO3AYZ9CAA"

access_token = "434947831-SFNowTmRAFbfgqBLe4pqbb7DwlxIcds25axbZZ2M"
access_token_secret = "Kjd8ukV9tdw4R9yWWhvK4gaX80zPnLUnMq0Bf57Y1jCQG"

bearer_token = "AAAAAAAAAAAAAAAAAAAAANORrwEAAAAAv1plFYlKqggQL9DIlTfJFmelyuc%3D6XdSjHENE5ovxgupTL6o4Wr654QXL2E8IgPcgqT3HzhqW9ZfIZ"

# write a function that takes a list consisting of an endpoint url, key and secret and stores it in a line in a file in csv format.
# if the file does not exists, create it and add the line to it.
# if the file exists, append the line to it and if the endpoint exists, overwrite it.
def configure_api_settings( sg ):
    # use the global value of api_key
    global api_key
    global api_key_secret
    global access_token
    global access_token_secret 
    if os.path.isfile('api_settings.csv'):
        with open('api_settings.csv', 'r') as f:
            reader = csv.reader(f)
            row = next(reader)
            api_key, api_key_secret, access_token, access_token_secret = row
    # get the api key values from the user
    new_api_key = sg.popup_get_text('Enter api key or leave blank for default')
    # check if the user entered a value, if so set the api key to the value
    if new_api_key != '' and new_api_key != None and new_api_key != ' ': 
        api_key = new_api_key
        new_api_key_secret = sg.popup_get_text('Enter api secret key or leave blank for default')
        if new_api_key_secret != '' and new_api_key_secret != None and new_api_key_secret != ' ':
            api_key_secret = new_api_key_secret
            new_access_token = sg.popup_get_text('Enter access token or leave blank for default')
            if new_access_token != '' and new_access_token != None and new_access_token != ' ':
                access_token = new_access_token
                new_access_token_secret = sg.popup_get_text('Enter access token secret or leave blank for default')
                if new_access_token_secret != '' and new_access_token_secret != None and new_access_token_secret != ' ':
                    access_token_secret = new_access_token_secret
    # create a list consisting of the api key values
    api_settings = [api_key, api_key_secret, access_token, access_token_secret ]
    # write the list to a file in csv format
    with open('api_settings.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(api_settings)
    print(api_key + "\n", api_key_secret +"\n", access_token + "\n", access_token_secret + "\n")
   


# write a function that uses tweepy to authenticate with the Twitter API using the key and secret
def authenticate_with_twitter_api():
    global api_key
    global api_key_secret
    global access_token
    global access_token_secret
    # create an OAuthHandler object using the key and secret
    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    # use the access token and access token secret to authenticate with the Twitter API
    auth.set_access_token(access_token, access_token_secret)
    # create an API object using the OAuthHandler object
    api = tweepy.API(auth)
    return api

def get_top_tweets(api, keywords):
    # get the tweets from the api search of the keywords

    tweets = []
    # get the tweets from the api search of the keyword
    #tweets += api.search_recent_tweets(q=ke
    # ywords, lang="en", count=10, tweet_mode="extended")
    tweets = tweepy.Cursor(api.search_tweets, q=keywords, lang="en", tweet_mode="extended").items(10)
    # convert the tweets to a pandas dataframe
    data = []
    for tweet in tweets:
        data.append({
            'id': tweet.id,
            'created_at': tweet.created_at,
            'text': tweet.full_text,
            'likes': tweet.favorite_count,
            'views': tweet.retweet_count
            # Add more fields as needed
        })

    df = pd.DataFrame(data)


    return tweets