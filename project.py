"""
project.py - run this to re-create my submission. 

    Note: the find_hyperparameters argument to model_and_submit() is set to 
    False, meaning that the RandomForestClassifier will use a set of 
    hyperparameters that have already been tuned. Set this argument to True to 
    use grid search to find the hyperparameters yourself. Takes ~ 40 min.
"""

import fancyimpute
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV


def ingest_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return pd.concat([train.assign(Train = 1), 
        test.assign(Train = 0).assign(Survived = -999)[list(train) + ['Train']]]
    )


extract_lastname = lambda x: x.split(',')[0]


def extract_title(x):
    """
    Get the person's title from their name. Combine reduntant or less common 
    titles together.
    """
    title = x.split(',')[1].split('.')[0][1:]
    if title in ['Mlle', 'Ms']:
        title = 'Miss'
    elif title == 'Mme':
        title = 'Mrs'
    elif title in ['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 
                   'the Countess', 'Jonkheer', 'Dona']:
        title = 'Esteemed'
    return title


first_letter = np.vectorize(lambda x: x[:1]) 


def create_dummy_nans(data, col_name):
    """
    Create dummies for a column in a DataFrame, and preserve np.nans in their 
    original places instead of in a separate _nan column.
    """
    deck_cols = [col for col in list(data) if col_name in col]
    for deck_col in deck_cols:
        data[deck_col] = np.where(
            data[col_name + 'nan'] == 1.0, np.nan, data[deck_col])
    return data.drop([col_name + 'nan'], axis = 1)


def impute(data):
    """
    Impute missing values in the Age, Deck, Embarked, and Fare features.
    """
    impute_missing = data.drop(['Survived', 'Train'], axis = 1)
    impute_missing_cols = list(impute_missing)
    filled_soft = fancyimpute.SoftImpute().complete(np.array(impute_missing))
    results = pd.DataFrame(filled_soft, columns = impute_missing_cols)
    results['Train'] = list(data['Train'])
    results['Survived'] = list(data['Survived'])
    assert results.isnull().sum().sum() == 0, 'Not all NAs removed'
    return results


def feature_engineering(data):
    return (data

        # Turn the Name feature into LastName, Title features
        .assign(LastName = lambda x: x.Name.map(extract_lastname))
        .assign(Title = lambda x: x.Name.map(extract_title))

        # Turn the Cabin feature into a Deck feature (A-G)
        .assign(Deck = lambda x: np.where(
            pd.notnull(x.Cabin), first_letter(x.Cabin.fillna('z')), x.Cabin))
        .assign(Deck = lambda x: np.where(x.Deck == 'T', np.nan, x.Deck))

        # Turn Sex into a dummy variable
        .assign(Sex = lambda x: np.where(x.Sex == 'male', 1, 0))

        # Create dummy variables for the categorical features - not stricly 
        # necessary for random forests, but useful to have if we want to compare
        # RF results to other algorithms
        .assign(Pclass = lambda x: x.Pclass.astype(str))
        .pipe(pd.get_dummies, columns = ['Pclass', 'LastName', 'Title'])
        .pipe(pd.get_dummies, columns = ['Deck'], dummy_na = True)
        .pipe(pd.get_dummies, columns = ['Embarked'], dummy_na = True)
        .pipe(create_dummy_nans, 'Deck_')
        .pipe(create_dummy_nans, 'Embarked_')

        # Drop columns we don't need
        .drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis = 1)

        # Impute NAs
        .pipe(impute)
    )


def split_data(data):
    """
    Split the combined training/prediction data into separate training and 
    prediction sets.
    """
    outcomes = np.array(data.query('Train == 1')['Survived'])
    train = (data.query('Train == 1')
             .drop(['Train', 'Survived'], axis = 1))
    to_predict = (data.query('Train == 0')
                  .drop(['Train', 'Survived'], axis = 1))
    return train, outcomes, to_predict


def train_test_model(model, hyperparameters, X_train, X_test, y_train, y_test,
                    folds = 5):
    """
    Given a [model] and a set of possible [hyperparameters], along with 
    matricies corresponding to hold-out cross-validation, returns a model w/ 
    optimized hyperparameters, and prints out model evaluation metrics.
    """
    optimized_model = GridSearchCV(model, hyperparameters, cv = folds, 
        n_jobs = -1)
    optimized_model.fit(X_train, y_train)
    predicted = optimized_model.predict(X_test)
    print 'Optimized parameters:', optimized_model.best_params_
    print 'Model accuracy:', optimized_model.score(X_test, y_test), '\n'
    return optimized_model


def create_submission(name, model, train, outcomes, to_predict):
    """
    Train [model] on [train] and predict the probabilties on [test], and
    format the submission according to Kaggle.
    """
    model.fit(np.array(train), outcomes)
    probs = model.predict(np.array(to_predict))
    results = pd.DataFrame(probs, columns = ['Survived'])
    results['PassengerId'] = list(pd.read_csv('data/test.csv')['PassengerId'])
    (results[['PassengerId', 'Survived']]
        .to_csv('submissions/' + name, index = False))
    return None


def model_and_submit(train, outcomes, to_predict, find_hyperparameters):
    """
    Use a random forest classifier to predict which passengers survive the 
    sinking of the Titanic and create a submission.
    """
    if find_hyperparameters:
        X_train, X_test, y_train, y_test = train_test_split(
            train, outcomes, test_size = 0.2, random_state = 50)
        param_grid = {'n_estimators': [10, 50, 100, 300, 500, 800, 1000],
                      'criterion': ['gini', 'entropy']}
        rf_model = train_test_model(
            RandomForestClassifier(), 
            param_grid, X_train, X_test, y_train, y_test)
        model = rf_model.best_estimator_
    else:
        model = RandomForestClassifier(max_features = None, 
            min_samples_split = 1, n_estimators = 10, max_depth = 7, 
            min_samples_leaf = 1, n_jobs = -1)
    create_submission('rf_submission.csv', model, train, outcomes, to_predict)
    return None


def main():
    data = ingest_data()
    data = feature_engineering(data)
    train, outcomes, to_predict = split_data(data)
    model_and_submit(train, outcomes, to_predict, find_hyperparameters = False)


if __name__ == '__main__':
    main()
