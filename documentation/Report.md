# Data Mining and Machine Learning Project Report

## Introduction
For this final project of the Data Mining and Machine Learning course of the autumn semester of 2022/2023, our goal was to use the knowledge obtained in class and through research to analyze and classify sentences in French according to their difficulty. With this, it is possible to help non-native speakers to predict the difficulty of a text in French and find examples of reading material that are appropriate depending on a person's level of understanding (`A1` to `C2`).

## First steps
In order to do so, we begin by downloading all the necessary data and material to build and train our models. The data used was found on the Kaggle Competition page, and is separated in `training_data.csv`, `unlabelled_test_data.csv` and  `sample_submission.csv`. These files contain the training set, in which we will build and train our models after a split between test and train data, the actual test data that we wish to classify after our models are complete and an example of how our results on the test data must be submitted to Kaggle, respectively.

With all the data needed, the next step we took was to download the basic packs needed for our analysis during the project, those being:

```ruby
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 import seaborn as sns
 %matplotlib inline
 sns.set_style("whitegrid")
```

## Checking the baseline
With the packages installed, we move on to check the value of our baseline in the `training_data.csv` in order to have a better understanding of our data. For this, we begin by splitting our data into `x_train`, `x_test`, `y_train` and `y_test`, being the `x` values the sentences in French and the `y` values their difficulty. Once that was done, we used two different methods to calculate the baseline. 

The first method used, was with the `Dummy Classifier`, which we set to use the most frequent values, fit on our `y_train` set and score on the `y_test`, with this we obtained a baseline of 0.1677. The second method was used to confirm the value found previously, and in it we used the `.value_counts()` command to know the values of each difficulty in the whole dataframe used and which had a bigger frequency, once we knew that, we divided the value of the most frequent difficulty by the total amount and obtained a value of 0.1694 for our baseline. 

For all this process we needed to import from sklearn:

```ruby
 from sklearn.model_selection import train_test_split
 from sklearn.dummy import DummyClassifier
```

## Creating the models
Once we knew the value of our baseline, we continue on to create our models. The models used to start our classification during this project were `Logistic Regression`, `K-Nearest Neighbors`, `Decision Tree` and `Random Forest`, when it comes for the text analysis, we did not use any sort of data cleaning or tokenization for the firsts models created, we simply used the `TF-IDF Vectorizer`. After doing this base work to have a better understanding of our data, we chose to use `Neural Networks` as our extra technique for classification and with it we also applied various techniques of text analysis to try to improve our results.

### Logistic Regression
For all the cases mentioned above, we begin our coding by downloading the necessary packages and language sets to use desired classification method and the necessary text analysis. Using the example of our `Logistic Regression Classifier`, the packages downloaded to read our data and create our model were:

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

After this crucial step, we begin to code our model by creating the `pipeline` that we will use to fit our train data and classify our sentences, for the `Logistic Regression`, we use:

```ruby
 tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))
 lr = LogisticRegression(solver='lbfgs', max_iter=10000, random_state=0)

 pipe = Pipeline([('vectorizer', tfidf),
                  ('classifier', lr)])
```

It is important to note that the random state used for the `Logistic Regression` was set to 0, and that our pipeline also includes the vectorizer chosen for this step of the project.

Once our `pipeline` is created and we can predict the values of `y` on `x_test`, we use the function we defined as `evaluate` to calculate our test accuracy, precision, recall, F1-score and to form our confusion matrix. This function was used for all our base classification methods and was defined as follows:

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

The next step required for our analysis with the `Logistic Regression` model, was to identify examples wrongly classified texts, for this we had to compare the values of `y_test` to `y_pred` if they were identical, it meant that the classification was accurate, while if they were not equal, we had a wrongly classified text. Some examples found were:

> C'est en décembre 1967, après bien des invectives au Parlement, que sa loi relative à la régulation des naissances, dite loi Neuwirth est votée : elle autorise la vente exclusive des contraceptifs en pharmacie sur ordonnance médicale, avec autorisation parentale pour les mineures

> Giscard va pourtant réussir à transformer ce revers en tremplin

> Un choix difficile mais important : le public français écoute souvent les professionnels de Cannes pour choisir le film qu'il va aller voir au cinéma.

> Le débat porte plutôt sur l'utilité d'une telle mesure.

To finish all the analysis required for the `Logistic Regression`, we implemented the created model in our `unlabelled_test_data.csv` to generate a .csv file in the same format as `sample_submission.csv`, for this we defined our `x` as the sentence column of our dataframe and used our `pipeline` to predict the values of `y`.

### K-Nearest Neighbors

Moving on to the next classification model, the beginning of our `K-Nearest Neighbor` classification is extremely similar to the `Logistic Regression`, with little differences such as the need to import the following packs, besides the ones mentioned before:

```ruby
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
```

Once all the necessary packages were imported, we follow to create our `pipeline` in the same form as we did previously, with the exception that for our `KNN` classification we do not set a random state, and we will begin our analysis without setting any parameters, so the code goes as show below

```ruby
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))
knn = KNeighborsClassifier()

pipe = Pipeline([('vectorizer',tfidf),
                 ('classifier', knn)])
```                 

After running our `pipeline` and obtaining the predictions of `y`, we run our `evaluate` function and notice that the results for accuracy, precision, recall and F1-score are low, and in order to improve those results, we tune our hyperparameters using `GridSearchCV()` to find the best possible parameters to run our classification with. In order to do so, we apply the following code:

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

By tuning the hyperparameters, we find that the best parameters for our `KNN` classification are: 
 - n_neighbors: 29 
 - p: 2 
 - weights: distance

Inserting those values in our classifier, we create a new pipeline, `pipekg` as shown on the code below, to improve our evaluation results.

```ruby
knn_gs = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)

pipekg = Pipeline([('vectorizer',tfidf),
                   ('classifier', knn_gs)])
```

To finish our analysis with the `K-Nearest Neighbors Classifier` we once again use the function `evaluate` to obtain the necessary information, and afterwards we apply our model to the `unlabelled_test_data.csv`, generating another submission .csv file with the predicted values of `y`.

### Decision Tree

When working with the `Decision Tree` model, we will follow a pattern very similar to the one in the `K-Nearest Neighbors` model, meaning that after we run our `pipeline` we will tune our parameters in order to improve our classification. In any case, we begin by importing the necessary commands:

```ruby
from sklearn.tree import DecisionTreeClassifier, plot_tree
```

With the new imports, we can create the necessary `pipeline` to begin our classification, noting that once again we set the random state of our classifier to 0 before using it.

```ruby
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))
dtc = DecisionTreeClassifier(random_state=0)

pipe = Pipeline([("vectorizer",tfidf),
                 ("classifier",dtc)])
```

After running our code, we are able to predict values for `y`, but after running the `evaluate` function, it is possible to observe that the results for accuracy, precision, recall and F1-score are not as high as expected.

Because of this, we need to find the maximum tree depth that will improve our results, to do so we needed to define the following functions:

```ruby
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
```

With this functions we are now able to run iterations on our code, searching to calculate our desired results for various `tree_depth` values, and with this array of results, we can proceed to plot a graph that relates the depth of our tree with the accuracy score, identifying the best possible value for our `tree_depth` variable.

```ruby
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
```

```ruby
plot_cross_validation_on_trees(depths=range(100,121), cv_scores_mean=test_accuracy_score, accuracy_scores=train_accuracy_score,title="Decision Tree",cv_scores_std=0)
max = pd.Series(test_accuracy_score).argmax()
pd.Series(test_accuracy_score).argmax(),pd.Series(test_accuracy_score).max()
```

Once we the code show above produces the necessary graph and returns the best value for `tree_depth`, we can finally create a new `pipelinedtc` to run our classification with.

```ruby
dtc = DecisionTreeClassifier(max_depth=100+max,random_state=0)

pipelinedtc = Pipeline([("tokenizer",tfidf),
                        ("classifier",dtc)])
```

And once again, the final steps for our `Decision Tree Classifier` is to run the `evaluate` function and apply our trained model to the `unlabelled_test_data.csv` and generate another set of predictions as a submissible .csv file.

### Random Forest

When we worked on our `Random Forest` classification, we followed the same patterns as before, beginning by importing the packages needed, which now include:

```ruby
from sklearn.ensemble import RandomForestClassifier
```

With the packages imported, we create our `pipeline` using the `Random Forest Classifier` with a random state of 0 as our classifier, and we still use the `TF-IDF Vectorizer`.

```ruby
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))
rf = RandomForestClassifier(random_state=0)

pipe = Pipeline([("vectorizer", tfidf),
                 ("classifier", rf)])
```

Once the prediction is made on our `x_test` we obtain the predicted values of `y` and we can then use the function `evaluate`, defined previously, to compare the obtained predictions to the actual `y_test` values. With this we obtain our accuracy, precision, recall, F1-score and confusion matrix, the only step to be taken after this is running our model on the `unlabelled_test_data.csv` in order to create a submittable .csv file.

### Neural Networks

For the proposed new model, we thought i would be intresting to evaluate how a neural network would perform on such text classification task, with the addition of a word better text embedding system. The famous package `TensowFlow` was used due to it`s more high level code usabillity, which allows the user to create the architecture of the neural netowork and train the model more easily.

Additionally, instead of using the `TFiIDF Vectorizer`, we used a multilingual word transformer from Google, which converts each sentence into a vector composed of 512 values, as seen on the example below:

```ruby
embed_1(["bounjour","je suis Victor"])

Output:
<tf.Tensor: shape=(2, 512), dtype=float32, numpy=
array([[ 0.15410367,  0.02259256, -0.0677164 , ...,  0.04396354,
        -0.00034411,  0.0091284 ],
       [ 0.05661425, -0.01597459, -0.04966924, ..., -0.0372095 ,
         0.0325578 , -0.02770256]], dtype=float32)>

```

For the data processing, it was necessary to encode the variables into number, so we used the scikitlearn `OrdinalEncoder` class and obtained the numerical equivalents of the CEFR on numbers ranging from 0 to 5. 

```ruby
oe = OrdinalEncoder()
oe.set_params(categories= [['A1', 'A2', 'B1',"B2","C1","C2"]])
v = y0.to_numpy().reshape(-1,1)
y0=pd.DataFrame(oe.fit_transform(v).reshape(4800),columns=["difficulty"])
print(y0.head())
oe.categories

Output:
difficulty
0         4.0
1         0.0
2         0.0
3         0.0
4         2.0

[['A1', 'A2', 'B1', 'B2', 'C1', 'C2']]

```
After the pre-processing, the data is split into training, validation and test data, using the below seen code to perform the split. 

```ruby

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=0) #test_size=0.5

X_train.shape,X_test.shape,X_valid.shape

Output:
((3840, 512), (384, 512), (576, 512))


```
For the traning procedure, the data is then subdivided into batches.

```ruby
def df_to_dataset(dataframe,y_t, shuffle=True, batch_size=30):
  df = dataframe.copy()
  #labels = df.pop('difficulty')
  df = df
  ds = tf.data.Dataset.from_tensor_slices((df, y_t))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds
  
bs = 5 # 15 foi um valor muito bom
x_tr = df_to_dataset(X_train,y_t=y_train,shuffle=True,batch_size = bs)
x_te = df_to_dataset(X_test,y_t=y_test,shuffle=True,batch_size = bs)
x_va = df_to_dataset(X_valid,y_t=y_valid,shuffle=True,batch_size = bs)
```

For the main Neural Network architecture we used a sequential model composed of regular neurons and dropout layers, followed by a final layer in which a sigmoid function is applied. 

```ruby
model = tf.keras.Sequential([
                            tf.keras.layers.Dense(1064,activation="relu"),
                            tf.keras.layers.Dropout(0.6),
                            tf.keras.layers.Dense(516,activation="relu"),
                            tf.keras.layers.Dropout(0.6),
                            tf.keras.layers.Dense(64,activation="relu"),
                            tf.keras.layers.Dropout(0.4),
                            tf.keras.layers.Dense(6,activation = "sigmoid") 
                            
                            ])
```
On the plots we can se the evolution of the accuracy score and the loss function over time, and it`s noticable that after 5 epochs the validation loss function reaches a plateau, but after the 14th iteration it starts to increase again.


![image](https://user-images.githubusercontent.com/99041142/209179649-3e8164db-49db-4f8a-aaec-93a2d52ccd36.png)

To generate the predictions of the test data we first need to convert the probability results generated by the sigmoid function on the last layer defining that the final label will be the label with the highest probability.



Finaly the main results can be generated by the evaluate function defined previously. 

```ruby
CONFUSION MATRIX:
[[52  5  4  1  1  1]
 [20 18 20  3  0  1]
 [ 2  8 41  9  2  2]
 [ 0  1  6 32  9  9]
 [ 3  1  1 15 31 20]
 [ 4  0  8 10 12 32]]
ACCURACY SCORE:
0.5365
CLASSIFICATION REPORT:
	Precision: 0.5372
	Recall: 0.5365
	F1_Score: 0.5254
```

## Results
|  | Logistic Regression | KNearestNeighbors | Decision Tree | Random Forest | Neural Networks |
| ------------- | ------------- | ------------- |------------- |------------- |------------- |
| Precision | 0.4340 | 0.3733	 | 0.3153 | 0.3968	 | 0.5372 |
| Recall  | 0.4354 | 0.3635	 | 0.3146 | 0.3937	| 0.5365 |
| F1-Score  | 0.4337 | 0.3419 | 0.3144	 | 0.3888	| 0.5254 |
| Accuracy  | 0.4354 | 0.3635	 | 0.3146	 | 0.3937 | 0.5365 |


## Conclusions
With the results presented above, we can clearly see that `Neural Networks Classifier` achieved the best values of precision, recall, F1-score and accuracy in the test set created in the `training_data.csv` used for this project.

This result was expected since, to use the `Neural Network Classifier` we cleaned our data, tokenized our text and also applied text embedding to achieve better quality in our classification.

Considering this results for the `training_data.csv`, and the fact that we took more time to prepare our data with the `Neural Networks` classification, we can assume that this classifier it will also have the best score when applied to our `unlabelled_test_data.csv`, and therefore the final submission made to the Kaggle competition page was generated with this model, in which we achieved a maximum score of 0.5300.

## Last steps
With all of our codes completed and after analyzing our results, the last thing necessary for the finalization of this project was to make a video explaining the process of development of all of our work. In _[nome do video](link do unlisted video)_, we start by talking about the problem presented, our goals and the algorithms that were used during the project. We also talk about our expected results and about the actual evaluation of our classifications.

After recording the video and doing the necessary editing, it was posted as an unlisted video on YouTube so it can only be accessed by the correct link, given in this text and below:

<div align="center">
  <a href="https://www.youtube.com/watch?v=FxyRRnbC9nk"><img src="https://img.youtube.com/vi/FxyRRnbC9nk/0.jpg" alt="Classification models for non-native French speakers according to CERR"></a>
</div>
