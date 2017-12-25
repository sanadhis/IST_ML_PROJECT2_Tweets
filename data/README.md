# Dataset of Words sorted by Their Frequencies
The `words-by-frequency.txt` is quick-and-dirty 125k-word dictionary from a small subset of Wikipedia.
The dictionary/dataset is used to perform infer_spaces function in order to split words that cannot be found in GloVe dictionary (especially hashtag), into N number of words. Example: *happylifeatschool* -> *happy life at school*.

Credits:
* http://stackoverflow.com/users/1515832/generic-human
* https://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words/11642687#11642687