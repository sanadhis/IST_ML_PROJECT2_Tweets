import re
import io
import pandas as pd

def remove_redundant_char(tweet):
    """Function to remove redundant characters in every word of each tweet.
       Example: Eveeeer -> Ever
    Args:
        tweet     (string) : A tweet with maximum length of 140 characters.
    Returns:
        tweet_new (string) : A tweet without repeating character in each of its word.
    """

    # init an empty new tweet
    tweet_new = ''
    
    # split the original tweet into words
    words = tweet.split()
    
    # Iterate through words
    for word in words:
        # Using regex, replace repeated chars in each word and form new tweet
        tweet_new += re.sub(r'([a-z])\1\1+$', r'\1 <elong>', word) + ' '
    
    return tweet_new.strip()

def append_sentiment(tweet):
    """Function to append sentiment for every positive/negative word of each tweet.
       Example: 
       Congratulation -> positive
       Failure -> negative
       Use opinion-lexicon-dataset (more details in README)
    Args:
        tweet     (string) : A tweet with maximum length of 140 characters.
    Returns:
        tweet_new (string) : A tweet with emphasized sentiment for its every positive/negative word.
    """

    # Load database of positive and negative words in English
    positive_words_dataset = './opinion-lexicon-english/positive-words.txt'
    negative_words_dataset = './opinion-lexicon-english/negative-words.txt'
    positive_words         = set(io.open(positive_words_dataset, encoding = "ISO-8859-1").read().split())
    negative_words         = set(io.open(negative_words_dataset, encoding = "ISO-8859-1").read().split())

    # init an empty new tweet
    tweet_new = ''

    # split the original tweet into words
    words = tweet.split()

    # Iterate through words
    for word in words:
        # Check each word and form new tweet with sentiment declaration
        if word in positive_words:
            tweet_new += 'positive '
        elif word in negative_words:
            tweet_new += 'negative '
        
        tweet_new += word + ' ' 

    return tweet_new.strip()

def expand_contraction(tweets):
    """Function to expand contraction (normalyy indicated with single quote) in indirect speech.
    Args:
        tweets     (panda series) : Tweets with maximum length of 140 characters each.
    Returns:
        tweets_new (panda series) : Tweets with expanded contractions for each contraction found.
    """

    # Define dictionary of english contractions: be, have, will, would
    english_contractions_dict = {
        'i\'m': 'i am',
        '\'re': ' are',
        'he\'s': 'he is',
        'what\'s': 'what is',
        'who\'s': 'who is',
        'when\'s': 'when is',
        'why\'s': 'why is',
        'how\'s': 'how is',
        'it\'s': 'it is',
        'that\'s': 'that is',
        'n\'t': ' not',
        '\'ll': ' will',
        '\'l': ' will',
        '\'ve': ' have',
        '\'d': ' would',
        '\'s': '',
        's\'': '',
    }

    # Compile regex for english contractions with or "|" statement
    pattern = re.compile('|'.join(english_contractions_dict.keys()))

    # Find the pattern and replace it with corresponding value
    tweets  = tweets.apply(lambda tweet: pattern.sub(lambda x: english_contractions_dict[x.group()], tweet))
    
    return tweets

def reduce_punctuation(tweets):
    """Function to reduce and emphasize repeated english punctuations.
    Args:
        tweets     (panda series) : Tweets with maximum length of 140 characters each.
    Returns:
        tweets_new (panda series) : Tweets with reduced and emphasized punctuations each.
    """
    
    # Check for punctuation and apply '<repeat>' if punctuations are repeated
    for punct in ['!', '?', '.']:
        regex  = "(\\"+punct+"( *)){2,}"
        tweets = tweets.str.replace(regex, punct+' <repeat> ', case=False)
    
    return tweets

def replace_digits(tweet):
    """Function to replace digits in each tweet.
    Args:
        tweet     (string)       : A tweet with maximum length of 140 characters.
    Returns:
        tweet_new (string)       : A tweet with digits replaced with '<number>'.
    """

    # init an empty new tweet    
    tweet_new = ''

    # split the original tweet into words
    words     = tweet.split()
        
    # Iterate through words
    for word in words:
        try:
            # Remove numerical symbols
            num = re.sub('[,\.:%_\-\+\*\/\%\_]', '', word)
            # If the word is number, replace with '<number>'
            float(num)
            tweet_new += '<number> '
        except:
            # if the word is not number, add without changing
            tweet_new += word + ' '
        
    return tweet_new.strip()

def emphasize_hashtag(tweet):
    """Function to emphasize hashtag for every hashtag found in each tweet.
    Args:
        tweet     (string)       : A tweet with maximum length of 140 characters.
    Returns:
        tweet_new (string)       : A tweet with emphasized hashtags.
    """

    # init an empty new tweet
    tweet_new = ''

    # split the original tweet into words
    words     = tweet.split()

    # Iterate through words
    for word in words:
        # add <hashtag> if hashtag is found
        if word.startswith("#"):
            tweet_new += '<hashtag> '
        tweet_new +=  word + ' '
        
    return tweet_new.strip()

def emoji_mapping(tweet, emoji_map):
    """Function to emphasize tweet sentiment based on common emojis.
    Args:
        tweet     (string)       : A tweet with maximum length of 140 characters.
        emoji_dic (panda series) : List of common emojis with category of sad, neutral, smile.
    Returns:
        tweet_new (string)       : A tweet with emphasized sentiment for each its emoji.
    """

    # init an empty new tweet
    tweet_new = ''

    # split the original tweet into words
    words     = tweet.split()
    
    # Iterate through words
    for word in words:
        # cannot find "<3" in GloVe embedding dataset
        # hence transform it into <heart>
        if word == "<3":        
           tweet_new += "<heart> "
        else:
            # try: if we can find the mapping in emoji_map, then use it
            # except: otherwise, use the same word
            try:
                emoji_map[word]
                tweet_new += emoji_map[word] + " "
            except:
                tweet_new += word + " "
    
    return tweet_new.strip()

def generate_emoji_maps(GloVe_tweet):
    """Function to generate the mapping between emoji and tags
    Args:
        GloVe_tweet (python dict)  : Pre-trained word vectors for twitter dataset using GloVe.
    Returns:
        emoji_dic   (panda series) : List of emojis and tags: <sadface>, <neutralface>, <smile>.
    """

    # Define emojis' components
    # possible combinations we observe in our training set
    eyebrows      = [">", ""]
    eyes          = [":", "8", ";", "="]
    noses         = ["-", "'", "o", ""]
    sad_mouth     = ["\\", "/", "[", "{"]
    neutral_mouth = ["", "o"]
    happy_mouth   = ["@", "]", "}", "d", "p", "q"]

    # Init empty panda series
    emoji_dict = pd.Series()

    for eyebrow in eyebrows:
        for eye in eyes:
            for nose in noses:
                for mouth in (sad_mouth + neutral_mouth + happy_mouth):
                    face = eyebrow + eye + nose + mouth

                    # try   : if we can find the emoji in our GloVe_tweet
                    #         then keep the original emoji
                    # except: if not, change emojis to the tags which we 
                    #         can find in GloVe_tweet
                    try:
                        GloVe_tweet.loc[face]
                        emoji_dict[face] = face
                    # Distinguish categories of each emoji based on mouth
                    except:
                        if mouth in sad_mouth:
                            emoji_dict[face] = "<sadface>"
                        elif mouth in neutral_mouth:
                            emoji_dict[face] = "<neutralface>"
                        else:
                            emoji_dict[face] = "<lolface>"

    return emoji_dict