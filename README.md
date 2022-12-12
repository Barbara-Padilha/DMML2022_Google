# DMML2022_Google

## Summary
* [Participants](https://github.com/Barbara-Padilha/DMML2022_Google#participants)
* [Project Objectives](https://github.com/Barbara-Padilha/DMML2022_Google#project-objectives)
* [Approach](https://github.com/Barbara-Padilha/DMML2022_Google#approach)
* [Results](https://github.com/Barbara-Padilha/DMML2022_Google#results)
* [YouTube Video](https://github.com/Barbara-Padilha/DMML2022_Google#youtube-video)

## Participants
 * Bárbara Maltoni Padilha
 * João Victor Velanes de Faria Ribeira da Silva
 
## Project Objectives
In this project, we had as a goal the classification of texts in French according to the level of proficiency it requires for a non-native french speaker to understand it (`A1`,`A2`,`B1`,`B2`,`C1` and `C2`).

In order to do so, we should use some of the methods tought during class, such as `Logistic Regression`, `K-Nearest Neighbors`, `Decision Trees` and `Random Forest`, along with the `TF-IDF Vectorizer`, to analyse sentences in a train set defined on the `training_data.csv` presented in the data file on this GitHub, without any data cleaning. After training our base models, we have to run our algorithms on the `unlabelled_test_data.csv` to generate the classification of the french sentences for submission. 

After this first step, we need to utilize a different classification method and text analysis techniques to try to obtain the best possible score for our text classification. 

## Approach
resumo de como fizemos

## Results
|  | Logistic Regression | KNearestNeighbors | Decision Tree | Random Forest | Neural Networks |
| ------------- | ------------- | ------------- |------------- |------------- |------------- |
| Precision | 0.4340 | 0.3733	 | 0.3153 | 0.3968	 | 0 |
| Recall  | 0.4354 | 0.3635	 | 0.3146 | 0.3938	| 0 |
| F1-Score  | 0.4337 | 0.3419 | 0.3144	 | 0.3888	| 0 |
| Accuracy  | 0.4354 | 0.3635	 | 0.3146	 | 0.3938 | 0 |

With the present results, we can cleary see that Neural Networks Classiffier achieved the best values of precision, recall, F1-score and accuracy in the test set created in the notebook used for this project.

This result was expected since, as described in the [Approach](https://github.com/Barbara-Padilha/DMML2022_Google#approach) section of this GitHub, to use the Neural Network Classifier we cleaned our data, tokenized our text and also applied text embeding to achieve better quality in our classification. Besides that, we also used a different vectorizer, that works better for the French language.

## YouTube Video
Video explaning the algorithms used this project, as well as an evaluation of the solutions obtained:

add link - unlisted video
