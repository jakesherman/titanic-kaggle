# titanic-kaggle

> The sinking of the RMS Titanic is one of the most infamous shipwrecks in history...In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to *predict which passengers survived the tragedy*.

This is an introductory Kaggle challenge where the goal is to predict which passengers survived the sinking of Titanic based on a set of attributes of the passengers including name, gender, age, and more.

## Feature engineering

After taking an initial stab at feature engineering, I took some ideas from [Megan Risdal](https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic/discussion) and [piplimoon](https://www.kaggle.com/piplimoon/titanic/leaderboard-0-8134). One of the fun parts about this challenge was seeing all of the really creative ideas that others have thought up. To summarise what I did:

* Extracted a person's title (Mr, Mrs, Miss, Col, etc.) from their name
* Created a family size feature by adding up the number of Siblings/Spouses and Parents/Children on board
* Created a family variable from people's last names and their family size - since some people share last names, last name + family size should be a good proxy for a specific family
* Used the ticket feature (where multiple people can share a ticket) only for cases where a ticket was shared by two or more people across the training and test sets (ths result is a bit of bleeding between the training and test sets)
* Figured out which deck a person's cabin was on from the cabin feature
* Used one-hot encoding to create dummies for categorical features
* Used the `fancyimpute` package to impute missing values using MICE

## Modeling

I used 5-fold grid search as part of a nested cross validation framework using the hold-out method to choose hyperparameters and do model selection. I tried logistic regression, KNN, random forest, SVM, and gradient boosted trees models. They all performed reasonably well (accuracy in the ~ .78 - .81 range) except for KNN. My best score on the public leaderboard was from creating a majority voting ensemble of the four reasonably well performing model but giving the random forest model 2 votes (out of 5), giving a score of ~ .825.

## To run

Uses Python 2.7, tested on Ubuntu 14.04 LTS.

```bash
python project.py --name <FILE-NAME>
```

Arguments:
* `--name` required, name of the resulting .csv file to create
* `--findhyperparameters` if you don't include this argument the script uses pre-optimized hyperparameters - including this argument results in grid search being used to optimize the hyperparameters. This takes ~ 1 - 1.5 hours depending on your machine.
