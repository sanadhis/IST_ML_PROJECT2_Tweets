# Project 2 of Machine Learning- 2017/2018

[CS-433 PCML](http://isa.epfl.ch/imoniteur_ISAP/!itffichecours.htm?ww_i_matiere=2217650315&ww_x_anneeAcad=2017-2018&ww_i_section=249847&ww_i_niveau=&ww_c_langue=en) - [EPFL](http://epfl.ch)

> Twitter Sentiment Analysis. The submission is made to competition platform [kaggle](https://www.kaggle.com/c/epfml17-text/).

## About the project
Through this project, we use **Long Short-Term Memory (LSTM) Neural Networks** with **Global Vectors for Word Representation (GloVe)** for sentiment analysis to decide the positive ":)" and negative ":(" tweets.

## Brief Overview
The main scripts for this project are models.py (LSTM Neural Networks) and run.py. We tried several classification algorithms such as Naive Bayes and Decision Tree, and our best accuracy is obtained by LSTM neural networks.
*The scripts for other ML implementations are not included in this project*.
* model.py : Scripts to train the model using training datasets and to produce and save our best models for tweets classifier system. The model will be stored to `/results/models` directory.
* run.py   : Script to produce our final submission to kaggle. The script will load the model via static assignment in `/results/models` and will produce submission file in `/results/submissions`.
<br /><br/>

**Note that all functions and helpers for the both scripts are stored in scripts/ directory, for more details you can go through README in scripts/ and read all methods directly**.

## Dependencies
To execute the models.py and run.py, the mininum requirements are as follows:
1. [Anaconda](https://www.anaconda.com/download/) with Python 3.

(Alternative) mininum installation (only pandas) in python 3:
```bash
$ pip install pandas
```
2. [Tensorflow](https://www.tensorflow.org/) (we use version of 1.3.0)

Installing in machine without gpu:
```bash
$ pip install tensorflow==1.3.0
```
<br>Installing in machine with gpu:
```bash
$ pip install tensorflow-gpu==1.3.0
```
3. Download the [Stanford](https://nlp.stanford.edu/projects/glove/)'s Pre-trained word Vectors for Twitter:
```bash
wget https://nlp.stanford.edu/data/glove.twitter.27B.zip -O glove-stanford/glove.twitter.27B.zip
unzip glove-stanford/glove.twitter.27B.zip -d glove-stanford/
```
4. Download [Twitter Datasets](https://storage.googleapis.com/kaggle-competitions-data/kaggle/7744/twitter-datasets.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1514109264&Signature=dUZ7wANahBZUosKsPftxGuQz6NWjaPjIdVKwazTP1vklm%2FGOZxs1nmzL73ApNO2xtlrM3qwsF6Zw0iBI%2FVfZUcUXm4GUG84Z6f%2FQhdXu52KFKlHXuo) from Kaggle:

**Please simply put the datasets (train_pos_full.txt, train_neg_full.txt and test_data.txt) in twitter-datasets/ directory.**
```bash
mv ~/Downloads/twitter-datasets.zip -O twitter-datasets/twitter-datasets.zip
unzip twitter-datasets/twitter-datasets.zip -d twitter-datasets/
```

## OS Environment
**Please run the scripts in unix-based operating system. We suggest to run our scripts on Linux machine (Ubuntu 14.04 or 16.04) or MacOSX operating system.**

# Project Structure
------------

    ├── models.py                      : Main script to produce our models using LSTM neural networks.
    ├── run.py                         : Main script to produce our submission using pre-trained models.
    ├── README.md                      : The README guideline and explanation for our project.
    |
    ├── data
    │   ├── words-by-frequency.txt     : Quick-and-dirty 125k-word dictionary from a small subset of Wikipedia.
    │   ├── README.md                  : Brief explanation for words-by-frequency dataset.
    |
    ├── glove-stanford
    │   ├── glove.twitter.27B.200d.txt : Pre-trained word vectors of twitter dataset by Stanford NLP group.
    │   ├── README.md                  : Brief explanation for the glove dataset file.
    |
    ├── opinion-lexicon-english
    │   ├── negative-words.txt         : Dataset of negative words in English.
    │   ├── positive-words.txt         : Dataset of positive words in English.
    │   ├── README.md                  : Brief explanation of the opinion-lexicon datasets.
    |
    ├── report                         : Directory containing our report file in LaTeX and pdf format.
    |
    ├── results
    │   ├── models                     : Directory containing pre-trained models from our training process, models are generated by models.py.
    │   ├── submissions                : Directory containing submission files, submission files are generated by run.py.
    |
    ├── scripts
    │   ├── glove_embeddings.py        : Script containing trick and magic to perform glove_embedding.
    │   ├── helpers.py                 : Script containing helpers method.
    │   ├── preprocessings.py          : Script containing our methods to preprocess tweets before performing the main training.
    │   ├── README.md                  : Brief explanation for glove_embeddings, helpers, and preprocessings scripts.
    |
    ├── twitter-datasets
    │   ├── test_data.txt              : Twitter test dataset, containing 10,000 unlabeled tweets.
    │   ├── train_neg_full.txt         : Twitter training dataset, containing 1,250,000 negative tweets.
    │   ├── train_pos_full.txt         : Twitter training dataset, containing 1,250,000 positive tweets.
    |


--------

## More on Technical Overview
### Data Preprocessing
* *Remove numbers*
<br>Removing numbers as they are less useful for sentiment analysis.
* *Remove repeating characters*
<br>Ensuring the basic form of each word for glove embedding.
* *Expand English contraction*
<br>Splitting the contraction in indirect speech (e.g I'm).
* *Reduce English punctuation*
<br>Reducing repetitive punctuations (e.g. !!).
* *Highlight sentiment words*
<br>Highlighting the presence of sentiment words, either positive or negative ones.
* *Split hashtags into words*
<br>Splitting unspaced sentence or phrase (hashtag #) and performing this right before procedding glove embedding, in order to maximize the number of words vector representations of each tweet.
* *Transform emojis into special words*
<br>Changing emojis into words for glove embedding.

### Glove Embedding
We use the GloVe embedding dataset for tweets from [Stanford NLP group](https://nlp.stanford.edu/projects/glove/): [glove.twitter.27B.200d.txt](http://nlp.stanford.edu/data/glove.twitter.27B.zip) <br>

Description of GloVe dataset:
> This dataset provides the mapping between frequently used words in tweets and 200-dimension word vectors. The words inside the dataset are varies in terms of languages, e.g. English, Chinese, Japanese, etc.

### Training
Each tweet is represented with a matrix of size 40x200, and the row vectors of the matrix represent the words in tweet.
* 10 epochs
* RMSProp with learning rate = 5e-4, decay = 0.9 and no momentum
* 1000 tweets in each step (2500 steps in each epoch)

## models.py - LSTM Neural Networks
Our final and best models can be reproduced by executing script models.py.

We use Long Short Term Memory networks with 1024 units trained over 40 timesteps. The LSTM takes in a glove embedded word at each timestep and outputs a 1024 dimensional feature vector at t=40. We then pass this vector through 4 fully connected layers with [512,512,512,2] where the final layer with 2 units compute the sigmoid operation.

**Note that it takes around *4 hours* to train our model in a machine with 2 x Intel Xeon E5-2680 v3 (Haswell), 256GB Memory, and 2x Nvidia Titan X (ICC.T3 in epfl iccluster).**

## run.py - Creating Final Submission File  
Our final result can be reproduced by executing script run.py.
<br />In run.py, **we use our best pre-trained model** to reproduce our final submission. 
* With GPU, run.py takes roughly around ~**8 minutes** of execution time to produce the submission file.
* Without GPU, run.py needs to run for ~**20 minutes** to produce the submission file.

* Public leaderboard
  - **88.040%** of accuracy.
* Private Leadeboard
  - **87.560%** of accuracy.

## How to use models.py

1. Ensure that you have python 3 in your machine.
2. Ensure that you have pandas, numpy, pickle (come along within Anaconda packages).
3. Ensure you have tensorflow (we use version 1.3.0).
4. To run the script, simply execute:

  ```bash
  $ python models.py
  ```

## How to use run.py

1. Ensure that you have python 3 in your machine.
2. Ensure that you have pandas, numpy, pickle (come along within Anaconda packages).
3. Ensure you have tensorflow (we use version 1.3.0).
4. To run the script, simply execute:

  ```bash
  $ python run.py
  ```

## Team - "Instant Noodles are the Best"
[Project Repository Page](https://github.com/sanadhis/IST_ML_PROJECT2_Tweets)
- Cheng-Chun Lee ([@wlo2398219](https://github.com/wlo2398219)) : (cheng-chun.lee@epfl.ch)
- Haziq Razali ([@haziqrazali](https://github.com/haziqrazali)) : (muhammad.binrazali@epfl.ch)
- Sanadhi Sutandi ([@sanadhis](https://github.com/sanadhis))    : (i.sutandi@epfl.ch)
