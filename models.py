# Import native python libary
import datetime

# Import conda-bundled libraries
import pandas as pd
import numpy  as np
import pickle

# Import tensorflow (we use version 1.3.0)
import tensorflow as tf

# Import helpers, our preprocessings, and glove embeddings scripts
from scripts.helpers          import create_csv_file, read_file, batch_iter
from scripts.preprocessings   import remove_redundant_char, append_sentiment
from scripts.preprocessings   import generate_emoji_maps, emoji_mapping
from scripts.preprocessings   import expand_contraction, reduce_punctuation
from scripts.preprocessings   import emphasize_hashtag, replace_digits
from scripts.glove_embeddings import glove_embeddings

# Static Vars
train_positive_tweets  = "./twitter-datasets/train_pos_full.txt"
train_negative_tweets  = "./twitter-datasets/train_neg_full.txt"
glove_file             = "./glove-stanford/glove.twitter.27B.200d.txt"
model_file             = "./results/models/model-" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# Load training dataset
tweets_pos = read_file(train_positive_tweets)
tweets_neg = read_file(train_negative_tweets)

# Add positve & negative columns: one-hot encoding
tweets_pos['positive'] = 1
tweets_pos['negative'] = 0
tweets_neg['positive'] = 0
tweets_neg['negative'] = 1

# Merge Positive and Negative Dataset
tweets_all = pd.concat([tweets_pos, tweets_neg])

# Load the GloVe embedding dataset got from Stanford NLP group
# Reference: https://nlp.stanford.edu/projects/glove/
print("Reading glove")
GloVe_tweet = dict()
fp = open(glove_file, 'r')
for line in fp:
    tokens = line.strip().split()
    GloVe_tweet[tokens[0]] = np.array([float(val) for val in tokens[1:]])
GloVe_tweet = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in GloVe_tweet.items() ])).transpose()

# Data preprocessing
tweets_all = tweets_all.reset_index(drop=True)
tweets_all['tweet'] = tweets_all.tweet.apply(lambda t: t.decode("utf-8"))

# thaaanks = thanks
print("Filtering Repeated chars")
tweets_all['tweet'] = tweets_all.tweet.apply(lambda t: remove_redundant_char(t))

# like -> positive, hate -> negative
print("Emphasized Sentiment words")
tweets_all['tweet'] = tweets_all.tweet.apply(lambda t: append_sentiment(t))

# change emojis to specific tags if we can't find them in our GloVe embedding dataset
print("Transform emoji")
emoji_map           = generate_emoji_maps(GloVe_tweet)
tweets_all['tweet'] = tweets_all.tweet.apply(lambda tweet: emoji_mapping(tweet, emoji_map))

# you're -> you are ; i'll -> i will
print("Expand English contraction")
tweets_all['tweet'] = expand_contraction(tweets_all.tweet)

# !!! -> ! <repeat>
print("Reduce English punctuation")
tweets_all['tweet'] = reduce_punctuation(tweets_all.tweet)

# Add <hastag> when we see hashtag, it increases the accuracy
# but so far we don't find any reason to support it
print("Emphasize hashtag")
tweets_all['tweet'] = tweets_all.tweet.apply(lambda t: emphasize_hashtag(t))

# Remove numbers
print("Filtering digits")
tweets_all['tweet'] = tweets_all.tweet.apply(lambda t: replace_digits(t))

# Define ratio: proportion of training sets
ratio = 0.9
N_data = tweets_all.shape[0]
N_train_data = (int)(N_data * ratio)

# get the shuffle indices 
indices_shuffle = np.random.permutation(np.arange(tweets_all.shape[0]))

indices_tr = indices_shuffle[0:N_train_data]
indices_te = indices_shuffle[N_train_data:]

# Define setting for glove embedding and model training
max_num_words = 40
embedding_size = 200
n_classes = 2
batch_size=1000

#  [batch_size, 40, 200]
x = tf.placeholder(tf.float32, [None, max_num_words, embedding_size], name='embedding') 
y = tf.placeholder(tf.float32, [None, n_classes], name='class_probability')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

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
lstm  = tf.nn.rnn_cell.LSTMCell(num_units = 1024, state_is_tuple=True)
lstm_state = lstm.zero_state(batch_size, tf.float32)

# lstm
for i in range(40):
    lstm_out, lstm_state = lstm(inputs = x[:,i,:], state = lstm_state)

# fc layers
out = tf.matmul(lstm_out, fc1_w) + fc1_b #lstm_out
out = tf.nn.dropout(out, keep_prob) 
out = tf.matmul(out, fc2_w) + fc2_b
out = tf.nn.dropout(out, keep_prob) 
out = tf.matmul(out, fc3_w) + fc3_b
out = tf.nn.dropout(out, keep_prob) 

# classification          
out = tf.matmul(out, clf_w) + clf_b
scores = tf.nn.softmax(out, name='predictions-probability')
y_pred = tf.argmax(scores, axis=1, name='predictions')

# compute loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y), name='loss')

correct_prediction = tf.equal(y_pred, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='accuracy')

optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-4).minimize(loss)

# initialize all variables
init = tf.global_variables_initializer()

# to save variables
saver = tf.train.Saver(max_to_keep=100)

# Start a new TF session
sess = tf.Session()

# Run the initializer
sess.run(tf.global_variables_initializer())

# Epochs
step = 0
tr_step = 0
te_step = 0

# used for plot the accuracy
# batch_accuracies_tr = []
# batch_accuracies_te = []

# run for 10 epochs
for i in range(1, 11):
    
    # training, with batch_size tweets every time (1000 by default)
    for indices in batch_iter(train_indices = indices_tr, batch_size = batch_size):
        
        step = step + 1
        tr_step = tr_step + 1
        
        # get batch of tweets
        batch_tweets = tweets_all.iloc[indices]
        
        # transform tweets to matrices
        x_tr_batch = glove_embeddings(batch_tweets, GloVe_tweet)
        y_tr_batch = np.array(batch_tweets[['positive', 'negative']])        
        _, batch_loss, batch_accuracy = sess.run([optimizer, loss, accuracy], feed_dict={x: x_tr_batch, y: y_tr_batch, keep_prob: 0.5})
        
        
        # print the accuracy after every 20 epochs
        if(tr_step%20==0):            
            print('Epoch ', i, ' - Step ', step, '- Loss: ', batch_loss, ' - Tr acc: ', batch_accuracy)

            # save the results for plot
            # batch_accuracies_tr.append(batch_accuracy)

    # validation
    for indices in batch_iter(train_indices = indices_te, batch_size = batch_size):
        
        te_step = te_step + 1
        
        batch_tweets = tweets_all.iloc[indices]
        x_te_batch   = glove_embeddings(batch_tweets, GloVe_tweet)
        y_te_batch   = np.array(batch_tweets[['positive', 'negative']])        
        batch_loss, batch_accuracy = sess.run([loss, accuracy], feed_dict={x: x_te_batch, y: y_te_batch, keep_prob: 1.0})
    
        if(te_step%20==0):
            print('Epoch ', i, ' - Step ', step, '- Loss: ', batch_loss, ' - Te acc: ', batch_accuracy)

            # save the results for plot
            # batch_accuracies_te.append(batch_accuracy)

    # save the model after each epoch
    save_path = saver.save(sess, model_file, i)


    # np.save('tr_acc', batch_accuracies_tr)
    # np.save('te_acc', batch_accuracies_te)