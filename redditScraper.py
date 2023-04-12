import praw
import pickle
import re
import os

#Below could be refactered for clarity
def retrieve_db(groupings, n_samples, recreate_db):
    pickle_lists = []
    i = 0
    if recreate_db:
        destroy_pickles(groupings)

    for group in groupings:
        pickle_lists.append([])#adds empty list to group lists of comments into categories of their groupings
        for subR in group:
            loc_path = subR + '.pickle'
            try:
                pickle = open_data(loc_path)
                pickle_lists[i].append(pickle)
                print(loc_path + ' found')

            except OSError:
                save_data(loc_path, subR, n_samples)
                pickle = open_data(loc_path)
                pickle_lists[i].append(pickle)
                print(loc_path + ' saved')

        i += 1

    return pickle_lists

#Bellow code returns error because for some reason it doesn't open the file as a list
def open_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_data(path, subR, n_samples):
    with open(path, 'wb') as f:
        pickle.dump(get_data(subR, n_samples), f)

def destroy_pickles(groupings):#so glad I got to name it this
    for group in groupings:
        for subR in group:
            try:
                loc_path = subR + '.pickle'
                os.remove(loc_path)
            except OSError:
                print("The file has already been deleted, or never existed")
                continue


# returns an list of comments
def get_data(subR, n_samples):
    count = 0
    data = []
    reddit = authenticate()
    posts = reddit.subreddit(subR).hot(limit = 1000) #1000 is max, use 1 - 10 for testing

    #iterates through first layer of replies to every comment to every collected post, and checks for varius conditions
    for post in posts:
        post.comments.replace_more(limit = 32) #requests and replaces up to 32 unloaded comments, remainder are removed
        print(count)
        for comment in post.comments:
            if count < n_samples:
                string = filter_data(comment)
                if string != None:
                    data.append(string)
                    count += 1

                #try commenting this bottom portion out to see if it improves model
                for reply in comment.replies:
                    if count < n_samples:
                        string = filter_data(reply)
                        if string != None:
                            data.append(string)
                            count += 1

            else:
                return data

    print('___WARNING,_ONLY_', count, '_COMMENTS_WERE_DISCOVERED_IN_', subR)
    return data

#returns authenticated instance of reddit
def authenticate():
    reddit = praw.Reddit(
        client_id = "#####################",
        client_secret = "#######################",
        password = "################",
        user_agent = "Web Scraper for Personal ML Project 1.0 by /u/###############",
        username = "###############",
    )
    return reddit

def filter_data(data):
    string = data.body
    string = string.lower()#converts to lowercase

    #removes URls, subreddits, and replaces punctuation with spaces to seperate words.
    string = re.sub(r'http\S+', '', string)
    string = re.sub(r'r/\S+', '', string)
    string = string.replace('.', ' ')
    string = string.replace('!', ' ')
    string = string.replace('?', ' ')
    string = string.replace('/', ' ')
    string = string.replace(',', ' ')

    #removes any remaining special characters
    whitelist = set('abcdefghijklmnopqrstuvwxyz ')
    string = ''.join(filter(whitelist.__contains__, string))

    # checks to see if comment is garbage/empty data
    if (string != '') and (string != 'removed') and (string != 'deleted') and ('i am a bot' not in string):
        return string
