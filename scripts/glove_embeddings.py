from   math import log
import numpy as np

def glove_embeddings(tweets, glove_words, num_words=40, embedding_size=200):
    """Function to form a matrix of words' vector mapping from each tweet using Glove.
    Args:
        tweets            (panda dataframe) : Dataframe of tweets with maximum length of 140 characters each.
        glove_words       (python dict)     : Twitter glove embeddings dictionary based on Stanford's glove twitter dataset.
    Returns:
        tweets_embeddings (numpy matrix)    : Matrix vector representation of words in each tweet with size of N_data x num_words x embedding_size
    """

    # Find the number of data
    N_data    = tweets.shape[0]
    
    # Init embedding matrix with size of N_data x num_words x embedding_size
    tweets_embeddings = np.zeros([N_data, num_words, embedding_size])

    # Iterate through each tweet
    for i, tweet in enumerate(tweets.tweet):

        # split the tweet into words
        words = tweet.split()

        # Init word index and embedding index     
        word_index      = 0
        embedding_index = 0
        
        # Loop until maximum num_words is reached
        for k in range(num_words):
            if k < len(words):
                # Access word by word in tweet
                word = words[word_index]
                
                # Try to map each word of every tweet
                try:
                    tweets_embeddings[i, embedding_index, :] = glove_words.loc[word]
                    word_index      += 1
                    embedding_index += 1
                
                # Except if word is not present in glove_words dict
                except:
                    # try to decompose the word into several words
                    # categorize word with hashtag (#)
                    if (not word.startswith("#")):
                        word = "#" + word

                    # Split the hashtag, this will also work for splitting word(s) without spaces into words
                    splitted_words = split_hashtag(word)

                    # Iterate through splitted hashtag
                    for splitted_word in splitted_words.split():
                        # Check if the word is not one character, or at least "a" or "i"
                        if((len(splitted_word) != 1) or (splitted_word == "a") or (splitted_word == "i")):
                            try:
                                # Map each word into vector based on glove dictionary
                                tweets_embeddings[i, embedding_index, :] = glove_words.loc[splitted_word]
                                embedding_index += 1
                            except:
                                continue

                    # Increment word index regardless of sucessfull embedding or not
                    word_index += 1

                    # Continue the loop
                    continue                  

    return tweets_embeddings

def split_hashtag(hashtag):
    """Function to split hashtag into several number of word(s).
       Example: 
       thumbgreenappleactiveassignmentweeklymetaphor
       thumb green apple active assignment weekly metaphor
    Args:
        hastag     (string) : A hashtag word.
    Returns:
        new_string (string) : String of word(s) from decomposed hashtag word.
    """

    # init an empty string
    new_string = ''

    # try to perform infer_spaces
    try: 
        new_string = infer_spaces(hashtag[1:]).strip()
    # if exception found, just return the word without the (#)
    except: 
        new_string = hashtag[1:]

    return new_string

def infer_spaces(s):
    """
        Hastag will be splitted based on most frequent (most commonly occur) word component in hashtag.
        The datasets of words sorted by frequncies can be found in /data dir.
        # Reference: (http://stackoverflow.com/a/11642687)
    """

    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    WORD_FREQUENCIES = './data/words-by-frequency.txt'
    
    # Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
    words    = open(WORD_FREQUENCIES).read().split()
    wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
    maxword  = max(len(x) for x in words)

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return " ".join(reversed(out))
