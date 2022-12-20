# Data Mining and Machine Learning Project Report

## Introduction
For this final project of the Data Mining and Machine Learning course of the autumn semester of 2022/2023, our goal was to use the knowledge obtained in class and through reasearch to analyse and classify sentences in French according to their difficulty. With this, it is possible to help non-ntive speakers to predict the difficulty of a text in French and find exemples of reading material that are appropriate depending on a persons level of understanding (`A1` to `C2`).

## First steps
In order to do so, we begin by downlading all the necessary data and material to build and train our models. The data used was found on the Kaggle Competition page, and is separated in `training_data.csv`, `unlabelled_test_data.csv` and  `sample_submission.csv`. These files contain the training set, in which we will build and train our models after a split between test and train data, the actual test data that we wish to classify after our models are complete and an exemple of how our results on the test data must be submitted to Kaggle, respectively.

With all the data needed, the first step we took was to download the basic packs needed for our analysis during the project, those being:

```ruby
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 import seaborn as sns
 %matplotlib inline
 sns.set_style("whitegrid")
```

## Checking the baseline
With packages installed, we move on to check the value of our baseline in the `training_data.csv` in order to have a better understanding of our data. For this, we begin by splitting our datas into `x_train`, `x_test`, `y_train` and `y_test`, once that was done we used two different methods to calculate the baseline. 

The first method used, was the `Dummy Classifier`, which we set to use the most frequent values, fit on our y_train set and scored on the y_test, with this we obtained a baseline of 0.1677. The second method was used to confirm the value found previously, and in it we used the `.value_counts()` command to know the values of each difficulty in the whole dataframe used and which had a bigger frequency, once we knew that, we divided the value of the most frequent difficulty by the total amount and obtained a value of 0.1694 for our baseline. 

For all of this process we needed to import from sklearn:

```ruby
 from sklearn.model_selection import train_test_split
 from sklearn.dummy import DummyClassifier
```

## Creating the models
Once we knew the value of our baseline, we continue on to create our models. The models used to start our classification during this project were `Logistic Regression`, `K-Nearest Neighbors`, `Decision Tree` and `Random Forest`, when it comes for the text analysis, we did not use any sort of data cleaning or tokenization for the models created, we simply used the `TF-IDF Vectorizer`. After doing this base work to have a better understanding of our data, we chose to use Neural Networks as our extra technique for classification and with it we also applied various techniques of text analysis in order to try to improve our results.

### Logistic Regression
For all the cases mentioned above, we begin our coding by downloading the necessary packages and language sets to use desired classification method and the necessary text analysis. Using the exemple of our `Logistic Regression Classifier`, the packages downloaded to read our data and create our model were:

```ruby
# Install and update spaCy
 !pip install -U spacy
 !python -m spacy download fr

# Import necessary packages
 import spacy
 from spacy import displacy
 from sklearn.feature_extraction.text import TfidfVectorizer

 from sklearn.pipeline import Pipeline
 import string
 from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
 from spacy.lang.fr import French
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
```

After this crutial step, we begin to code our model by creating the `Pipeline` that we will to fit our train data and classify our sentences, for the `Logistic Regression`, we use:

```ruby
 tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))
 lr = LogisticRegression(solver='lbfgs', max_iter=10000, random_state=0)

 pipe = Pipeline([('vectorizer', tfidf),
                  ('classifier', lr)])
```

It is important to note that the random state used for the Logistic Regression was set to 0, and that our pipeline also includes the vectorizer chosen for this step of the project.

Once our `Pipeline` is created and we can predict the values of `y` on `x_test`, we use the function we defined as `evaluate` to calculate our test accuracy, precision, recall, F1-score and to form our confusion matrix. This function was used for all our base classification methods and was defined as follows:

```ruby
def evaluate(true, pred):
    global precision,recall,f1
    precision = precision_score(true, pred, average='weighted')
    recall = recall_score(true, pred, average='weighted')
    f1 = f1_score(true, pred, average='weighted')
    print(f"CONFUSION MATRIX:\n{confusion_matrix(true, pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(true, pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n\tPrecision: {precision:.4f}\n\tRecall: {recall:.4f}\n\tF1_Score: {f1:.4f}")
```

The next step required for our analysis with the `Logistic Regression` model, was to identify exemples wrongly classified texts, for this we had to compare the values of `y_test` to `y_pred` if they were identical, it meant that the classification was accurate, while if they were not equal, we had a wrongly classified text. Some exemples found were:

> C'est en décembre 1967, après bien des invectives au Parlement, que sa loi relative à la régulation des naissances, dite loi Neuwirth est votée : elle autorise la vente exclusive des contraceptifs en pharmacie sur ordonnance médicale, avec autorisation parentale pour les mineures

> Giscard va pourtant réussir à transformer ce revers en tremplin

> Un choix difficile mais important : le public français écoute souvent les professionnels de Cannes pour choisir le film qu'il va aller voir au cinéma.

> Le débat porte plutôt sur l'utilité d'une telle mesure.

To finish all the analysis required for the `Logistic Regression`, we implemented the created model in our `unlabelled_test_data.csv` to generate a a csv file in the same format as `sample_submission.csv`, for this we defined our `x` as the sentence column of our dataframe and used our `pipeline` to predict the values of `y`.

### K-Nearest Neighbors


### Decision Tree


### Random Forest


### Neural Networks
