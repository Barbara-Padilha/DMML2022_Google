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
```ruby
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
```

```ruby
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))
knn = KNeighborsClassifier()

pipe = Pipeline([('vectorizer',tfidf),
                 ('classifier', knn)])
```                 

```ruby
k_range = list(range(1,31,2))
parameters = { 'classifier__n_neighbors' : k_range,
               'classifier__p' : (1,2),
               'classifier__weights' : ['uniform','distance']
              }

gs = GridSearchCV(pipe, parameters, scoring='accuracy', return_train_score = False, verbose=1)
grid_search = gs.fit(x_train,y_train)
best_params = grid_search.best_params_

k = best_params['classifier__n_neighbors']
p = best_params['classifier__p']
w = best_params['classifier__weights']
```

Fitting 5 folds for each of 60 candidates, totalling 300 fits
By tuning the hyper parameters, we find that the best parameters for our KNN classification are: 
 n_neighbors: 29 
 p: 2 
 weights: distance

```ruby
knn_gs = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)

pipekg = Pipeline([('vectorizer',tfidf),
                   ('classifier', knn_gs)])
```


### Decision Tree

from sklearn.tree import DecisionTreeClassifier, plot_tree

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))
dtc = DecisionTreeClassifier(random_state=0)

pipe = Pipeline([("vectorizer",tfidf),
                 ("classifier",dtc)])

def run_cross_validation_on_trees(X, y, tree_depths, cv=5, scoring='accuracy'):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    for depth in tree_depths:
        tree_model = Pipeline([("tokenizer",tfidf),("decision_tree_classifier",DecisionTreeClassifier(max_depth=depth))])
        cv_scores = cross_val_score(tree_model, X, y, cv=cv, scoring=scoring)
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(X, y).score(X, y))
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    return cv_scores_mean, cv_scores_std, accuracy_scores

def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title):
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
    #ax.fill_between(depths, cv_scores_mean-2*cv_scores_std, cv_scores_mean+2*cv_scores_std, alpha=0.2)
    ylim = plt.ylim()
    ax.plot(depths, accuracy_scores, '-*', label='train accuracy', alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Tree depth', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    #ax.set_ylim(ylim)
    ax.set_xticks(depths)
    ax.legend()


test_accuracy_score  = []
train_accuracy_score = []
for i in range(100,121):
    pipelinedt = Pipeline([("tokenizer",tfidf),
                         ("decision_tree_classifier",DecisionTreeClassifier(max_depth=i,random_state=0))])
    pipelinedt.fit(x_train,y_train)
    tree_predictions = pipelinedt.predict(x_test)
    tree_predictions_train = pipelinedt.predict(x_train)
    test_accuracy_score.append(accuracy_score(y_test,tree_predictions))
    train_accuracy_score.append(accuracy_score(y_train,tree_predictions_train))

plot_cross_validation_on_trees(depths=range(100,121), cv_scores_mean=test_accuracy_score, accuracy_scores=train_accuracy_score,title="Decision Tree",cv_scores_std=0)
max = pd.Series(test_accuracy_score).argmax()
pd.Series(test_accuracy_score).argmax(),pd.Series(test_accuracy_score).max()

dtc = DecisionTreeClassifier(max_depth=100+max,random_state=0)

pipelinedtc = Pipeline([("tokenizer",tfidf),
                        ("classifier",dtc)])

### Random Forest

from sklearn.ensemble import RandomForestClassifier

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))
rf = RandomForestClassifier(random_state=0)

pipe = Pipeline([("vectorizer", tfidf),
                 ("classifier", rf)])

### Neural Networks


## Results
|  | Logistic Regression | KNearestNeighbors | Decision Tree | Random Forest | Neural Networks |
| ------------- | ------------- | ------------- |------------- |------------- |------------- |
| Precision | 0.4340 | 0.3733	 | 0.3153 | 0.3968	 | 0 |
| Recall  | 0.4354 | 0.3635	 | 0.3146 | 0.3938	| 0 |
| F1-Score  | 0.4337 | 0.3419 | 0.3144	 | 0.3888	| 0 |
| Accuracy  | 0.4354 | 0.3635	 | 0.3146	 | 0.3938 | 0 |


## Conclusions
With the present results, we can cleary see that Neural Networks Classiffier achieved the best values of precision, recall, F1-score and accuracy in the test set created in the notebook used for this project.

This result was expected since, as described in the  section of this GitHub, to use the Neural Network Classifier we cleaned our data, tokenized our text and also applied text embeding to achieve better quality in our classification.

Considering this results for the `training_data.csv`, and the fact that we took more time to prepare our data with the Neural Networks classification, we can assume that this classifier it will also have the best score when applied to our `unlabelled_test_data.csv`.

## Last steps
With all of our codes completed and after analysing our results, the last thing necessary for the finalization of this project was to make a video explaining the process of development of all of our work. In _[nome do video](link do unlisted video)_, we start by talking about the problem presented, our goals and the algorithms that were used during the project. We also talk about our expected results and about the actual evaluation of our classifications.

After recording the video and doing the necessary editing, it was posted as an unlisted video on YouTube so it can only be acessed by the correct link, given in this text and below:

##### TEST CHANGE VIDEO LATER
<div align="center">
  <a href="https://www.youtube.com/watch?v=lD3s3jsw3pc"><img src="https://img.youtube.com/vi/lD3s3jsw3pc/0.jpg" alt="we found a message... | Raft"></a>
</div>