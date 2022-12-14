# DMML2022_Google

## Summary
* [Participants](https://github.com/Barbara-Padilha/DMML2022_Google#participants)
* [Project Objectives](https://github.com/Barbara-Padilha/DMML2022_Google#project-objectives)
* [Approach](https://github.com/Barbara-Padilha/DMML2022_Google#approach)
* [Results](https://github.com/Barbara-Padilha/DMML2022_Google#results)
* [YouTube Video](https://github.com/Barbara-Padilha/DMML2022_Google#youtube-video)

## Participants
 * Bárbara Maltoni Padilha
 * João Victor Velanes de Faria Ribeiro da Silva
 
## Project Objectives

In this project, we had as a goal the classification of texts in French according to the level of proficiency it requires for a non-native French speaker to understand it (`A1`,`A2`,`B1`,`B2`,`C1` and `C2`).

In order to do so, we should use some of the methods taught during class, such as `Logistic Regression`, `K-Nearest Neighbors`, `Decision Trees` and `Random Forest`, along with the `TF-IDF Vectorizer`, to analyze sentences in a train set defined on the `training_data.csv` presented in the data file on this GitHub, without any data cleaning. After training our base models, we have to run our algorithms on the `unlabelled_test_data.csv` to generate the classification of the French sentences for submission. 

After this first step, we need to utilize a different classification method and text analysis techniques to try to obtain the best possible score for our text classification. 
 
## Approach
We began our project by importing all the necessary packages to generate our models, read our texts and also split the `training_data.csv` into `x_train`, `x_test`, `y_train` and `y_test`, being that `x` is the sentences in French that we wish to classify and `y` is the difficulty, therefore, the categories we wish to classify our texts into. It is into this data that we will create and train all our models in the beginning.

Afterwards, we move on to creating the necessary code for the basic classification methods show in class, which were `Logistic Regression`, `K-Nearest Neighbors`, `Decision Tree` and `Random Forest`, once these classifiers were paired with the `Tf-IDF Vectorizer` in their respective `pipelines` it was possible to generate the prediction of the difficulty of the French sentences used to train our model. In every case mentioned above we used a function called `evaluate` to obtain the value of the accuracy score, the precision, the recall, the F1-score, as well as the confusion matrix for each model.

There were slight differences in the required actions of each type of classification:
- in the `Logistic Regression` we had to locate examples of wrongly classified texts;
- in the `K-Nearest Neighbors` and in the `Decision Tree` we had to tune in our hyperparameters to improve our classification;
- in the `Logistic Regression`, `Decision Tree` and `Random Forest` we had to set the random state of our classifier to 0, which was not necessary in the `K-Nearest Neighbors` since it does not have this parameter.

After these codes were complete, we moved on to making our new model utilizing `Neural Networks`, in which we went through the process of using text embedding techniques, applying the `TensorFlow` package. We substituted the `TF-IDF Vectorizer` for a multilingual word embedding transformer from Google, which genrated vectors containing 512 values that became the features used to train the model, all this with the goal of obtaining better results for the accuracy, precision, recall and F1-score.

Once all our models were created and trained, we used them on the `unlabelled_test_data.csv` to generate .csv files in the same format as `submission_exemple.csv` so that they could be submitted to the Kaggle competition page.

The entire process of this project is better described in the [Report.md](https://github.com/Barbara-Padilha/DMML2022_Google/blob/main/documentation/Report.md) file in the `documentation` folder, and all the codes used are presented in the `code` folder, which include separate collab files for each model created.

## Results
|  | Logistic Regression | KNearestNeighbors | Decision Tree | Random Forest | Neural Networks |
| ------------- | ------------- | ------------- |------------- |------------- |------------- |
| Precision | 0.4340 | 0.3733	 | 0.3153 | 0.3968	 | 0.5372 |
| Recall  | 0.4354 | 0.3635	 | 0.3146 | 0.3937	| 0.5365 |
| F1-Score  | 0.4337 | 0.3419 | 0.3144	 | 0.3888	| 0.5254 |
| Accuracy  | 0.4354 | 0.3635	 | 0.3146	 | 0.3937 | 0.5365 |

With the present results, we can clearly see that Neural Networks Classifier achieved the best values of precision, recall, F1-score and accuracy in the test set created in the notebook used for this project.

This result was expected since, as described in the [Approach](https://github.com/Barbara-Padilha/DMML2022_Google#approach) section of this GitHub, to use the Neural Network Classifier we applied text embedding and created a network with many layers that, according to our results, managed to better grasp the profiling of the senteneces. 

Considering these results for the `training_data.csv`, and the fact that we took more time to prepare our data with the Neural Networks classification, we can assume that this classifier it will also have the best score when applied to our `unlabelled_test_data.csv`.

## YouTube Video
Video explaining the algorithms used this project, as well as an evaluation of the solutions obtained:

[Classification models for non-native French speakers according to CEFR](https://youtu.be/FxyRRnbC9nk)
<div align="center">
  <a href="https://youtu.be/FxyRRnbC9nk"><img src="https://img.youtube.com/vi/FxyRRnbC9nk/0.jpg" alt="Classification models for non-native French speakers according to CEFR"></a>
</div>
