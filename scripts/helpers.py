import io
import csv
import numpy as np
import pandas as pd

def read_file(filepath):
    """Function to generate dataframe of tweets from tweet dataset.
    Args:
        filepath (string)         : Directory path of tweet dataset file.
    Returns:
        tweets   (panda dataframe): Dataframe containing all tweets in file.
    """

    # Init an empty list
    tweets       = []

    # Read file
    tweets_file  = io.open(filepath, 'rb')

    # Iterate through lines in file
    for line in tweets_file:
        # Remove "\n" or " " in the end of every line.
        tweet = line.strip()

        # Append tweet line by line
        tweets.append(tweet)
    
    # Form the dataframe of tweets
    tweets = pd.DataFrame(tweets, columns=['tweet'])

    return tweets

def batch_iter(train_indices, batch_size, shuffle=True):
    """Function to generate the iterable set of indices with size of batch_size.
    Args:
        train_indices (numpy array): Matrix of indices of selected datapoint from training set.
        batch_size    (int)        : The size of each batch indices.
    Returns:
        itteratable   (numpy array): batch indices.
    """

    # Get the number of data
    n_data = len(train_indices)
    
    # shuffle indices
    if shuffle:
        shuffled_indices = np.random.permutation(train_indices)
    else:
        shuffled_indices = train_indices

    # Iterate and return object for (n_data/batch_size) times.
    for batch_num in range(int(np.ceil(n_data/batch_size))):
        start_index = batch_num * batch_size
        end_index   = min((batch_num + 1) * batch_size, n_data)
        if start_index != end_index:
            yield shuffled_indices[start_index:end_index]

def create_csv_file(predictions, filename):
    """Function to Creates the final submission file to Kaggle.
    Args:
        predictions (numpy array): Classification results of tweets with value of either 1 or -1 each.
        filename    (string)     : Name for submission file.
    Returns:
        None
    """

    # Open file
    with open('results/submissions/' + filename, 'w') as prediction_file:
        # Define header
        fieldnames = ['Id', 'Prediction']

        # Create csv writer object and write the header
        writer = csv.DictWriter(prediction_file, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()

        # Loop through prediction results and write each into row
        for idx, prediction in enumerate(predictions):
            writer.writerow({'Id':idx+1,'Prediction':prediction})
