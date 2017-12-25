# Import native python libary
import re
import datetime

# Import conda-bundled libraries
import pandas as pd
import numpy  as np
import pickle

# Import tensorflow (we use version 1.3.0)
import tensorflow as tf

# Import helpers, our preprocessings, and glove embeddings scripts
from scripts.helpers          import create_csv_file, read_file
from scripts.preprocessings   import remove_redundant_char, append_sentiment
from scripts.preprocessings   import generate_emoji_maps, emoji_mapping
from scripts.preprocessings   import expand_contraction, reduce_punctuation
from scripts.preprocessings   import emphasize_hashtag, replace_digits
from scripts.glove_embeddings import glove_embeddings

# Static Vars
test_dataset    = "./twitter-datasets/test_data.txt"
glove_file      = "./glove-stanford/glove.twitter.27B.200d.txt"
best_model      = "./results/models/best"
submission_file = "submission-" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".csv"

# Load test Dataset
tweets_all = read_file(test_dataset)

# Load the GloVe embedding dataset got from Stanford NLP group
# Reference: https://nlp.stanford.edu/projects/glove/
print("1. Building Glove Dictionary")
GloVe_tweet = dict()
fp = open(glove_file, 'r')
for line in fp:
    tokens = line.strip().split()
    GloVe_tweet[tokens[0]] = np.array([float(val) for val in tokens[1:]])
GloVe_tweet = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in GloVe_tweet.items() ])).transpose()

# Data Preprocessing
print("2. Data Preprocessing")
tweets_all['tweet'] = tweets_all.tweet.apply(lambda t: t.decode("utf-8"))
tweets_all['tweet'] = tweets_all.tweet.apply(lambda t: re.sub(r"([-+]?\d+),",'',t))

# thaaanks = thanks
print("->Filtering Repeated chars")
tweets_all['tweet'] = tweets_all.tweet.apply(lambda t: remove_redundant_char(t))

# like -> positive, hate -> negative
print("->Emphasized Sentiment words")
tweets_all['tweet'] = tweets_all.tweet.apply(lambda t: append_sentiment(t))

# change emojis to specific tags if we can't find them in our GloVe embedding dataset
print("->Transform emoji")
emoji_map           = generate_emoji_maps(GloVe_tweet)
tweets_all['tweet'] = tweets_all.tweet.apply(lambda tweet: emoji_mapping(tweet, emoji_map))

# you're -> you are ; i'll -> i will
print("->Expand English contraction")
tweets_all['tweet'] = expand_contraction(tweets_all.tweet)

# !!! -> ! <repeat>
print("->Reduce English punctuation")
tweets_all['tweet'] = reduce_punctuation(tweets_all.tweet)

# Add <hastag> when we see hashtag, it increases the accuracy
# but so far we don't find any reason to support it
print("->Emphasize hashtag")
tweets_all['tweet'] = tweets_all.tweet.apply(lambda t: emphasize_hashtag(t))

# Remove numbers
print("->Filtering digits")
tweets_all['tweet'] = tweets_all.tweet.apply(lambda t: replace_digits(t))

# Define setting for glove embedding and model training
print("3. Assign tensorflow settings and load the best model")
max_num_words  = 40
embedding_size = 200
n_classes      = 2
batch_size     = 1000

#  [batch_size, 40, 200]
x = tf.placeholder(tf.float32, [None, max_num_words, embedding_size], name='embedding') 
y = tf.placeholder(tf.float32, [None, n_classes], name='class_probability')

# weights - fc
fc1_w = tf.get_variable("fc1_w", shape=[1024, 512])
fc2_w = tf.get_variable("fc2_w", shape=[512, 512])
fc3_w = tf.get_variable("fc3_w", shape=[512, 512])
clf_w = tf.get_variable("clf_w", shape=[512, 2])

fc1_b = tf.get_variable("fc1_b", shape=[512])
fc2_b = tf.get_variable("fc2_b", shape=[512])
fc3_b = tf.get_variable("fc3_b", shape=[512])
clf_b = tf.get_variable("clf_b", shape=[2])

# weights - lstm 
lstm       = tf.nn.rnn_cell.LSTMCell(num_units = 1024, state_is_tuple=True)
lstm_state = lstm.zero_state(batch_size, tf.float32)

# lstm
for i in range(max_num_words):
    lstm_out, lstm_state = lstm(inputs = x[:,i,:], state = lstm_state)

# fc layers
out = tf.matmul(lstm_out, fc1_w) + fc1_b 
out = tf.matmul(out, fc2_w) + fc2_b 
out = tf.matmul(out, fc3_w) + fc3_b

# classification          
out    = tf.matmul(out, clf_w) + clf_b
scores = tf.nn.softmax(out, name='predictions-probability')
y_pred = tf.argmax(scores, axis=1, name='predictions')

# Start a new TF session and restore the best models
sess  = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, best_model)

# Feedforward
print("4. Begin Feedforward")
predictions_list = []
for i in range(10):
    tweet_batch = tweets_all[i*1000:(i+1)*1000]
    x_tr_batch  = glove_embeddings(tweet_batch, GloVe_tweet)      
    predictions = sess.run(y_pred, feed_dict={x: x_tr_batch})
    predictions[predictions==0] = -1
    predictions_list.append(predictions)

print("5. Finish Feedforward")

print("6. Make submission file")
# Making submission file
predictions_final = np.array(predictions_list)
predictions_final = np.reshape(predictions_final, newshape=[10000])

create_csv_file(predictions_final,submission_file)
print("7. Finished")
