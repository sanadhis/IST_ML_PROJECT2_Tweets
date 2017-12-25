## Scripts
All scripts with functions and helpers to perform our machine learning project.

## glove_embeddings.py
Contains the three functions related to creating vector representations of words from every tweet using GloVe embeddings.
1. glove_embeddings : (mapping each word in every tweet to vector representation)
2. split_hashtag : (splitting words, especially hashtag# that do not available in glove dictionary dataset)
3. infer_spaces : (dynamic programming to split word into words based on frequency of occurence)

## helpers.py
contains the three helper functions:
1. read_file : (reading datasets file)
2. batch_iter : (batch iterator used when training the model)
3. create_csv_file : (creating submission file with csv format, containing the predictions of tweets and their corresponding id)

## preprocessings.py
contains our functions for tweets preprocessing:
1. remove_redundant_words : (remove redundant char in each word to form the original word)
2. append_sentiment : (emphasize the presence of positive/negative words in each tweet)
3. expand_contraction : (expand english contraction in indirect speech)
4. reduce_punctuation : (reduce English speech punctuation)
5. replace_digits : (remove digits with empty string)
6. emphasize_hashtag : (emphasize the presence of hashtag)
7. emoji_mapping : (map the emoji into word; sad, neutral, lol)
8. generate_emoji_maps: (generate emoji mapping, especially when it is not available in glove dictionary)
