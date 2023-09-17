import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# def build_random_forest(train_file, test_file):
#     # Load the training data
#     train_df = pd.read_csv(train_file)
#
#     # Drop irrelevant features
#     train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
#
#     # Impute missing values
#     train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())
#     train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
#
#     # Convert categorical variables into numerical ones
#     train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'])
#
#     # Split the data into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(train_df.drop('Survived', axis=1), train_df['Survived'], test_size=0.3, random_state=42)
#
#     # Train a random forest classifier
#     rf = RandomForestClassifier(random_state=42)
#     rf.fit(X_train, y_train)
#
#     # Tune the hyperparameters using grid search
#     param_grid = {
#         'n_estimators': [100, 200, 500],
#         'max_depth': [5, 10, 20],
#         'min_samples_split': [2, 5, 10]
#     }
#     grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5)
#     grid_search.fit(X_train, y_train)
#     print('Best hyperparameters:', grid_search.best_params_)
#
#     # Evaluate the performance of the final model on the validation set
#     y_pred = grid_search.predict(X_val)
#     print('Accuracy on validation set:', accuracy_score(y_val, y_pred))
#
#     # Load the test data
#     test_df = pd.read_csv(test_file)
#
#     # Drop irrelevant features
#     test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
#
#     # Impute missing values
#     test_df['Age'] = test_df['Age'].fillna(train_df['Age'].mean())
#     test_df['Fare'] = test_df['Fare'].fillna(train_df['Fare'].mean())
#
#     # Convert categorical variables into numerical ones
#     test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'])
#
#     # Make predictions on the test data
#     y_pred_test = grid_search.predict(test_df)
#
#     # Save the predictions to a CSV file
#     submission_df = pd.read_csv('gender_submission.csv')
#     submission_df['Survived'] = y_pred_test
#     submission_df.to_csv('submission.csv', index=False)
#
#     return grid_search

# def feature_engineering(train, test):
#     # Combine train and test data for consistent feature engineering
#     all_data = pd.concat([train, test], axis=0, ignore_index=True)
#
#     # Create a new feature called FamilySize by adding the SibSp and Parch features
#     all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
#
#     # Binning the Age feature
#     bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
#     labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80']
#     all_data['AgeGroup'] = pd.cut(all_data['Age'], bins=bins, labels=labels, include_lowest=True)
#
#     # Binning the Fare feature
#     bins = [0, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500]
#     labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-100', '100-200', '200-300', '300-400', '400-500']
#     all_data['FareGroup'] = pd.cut(all_data['Fare'], bins=bins, labels=labels, include_lowest=True)
#
#     # Extracting titles from the Name feature
#     all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
#
#     # Replace rare titles with 'Other'
#     title_counts = all_data['Title'].value_counts()
#     rare_titles = title_counts[title_counts < 10].index
#     all_data['Title'] = all_data['Title'].replace(rare_titles, 'Other')
#
#     # Drop irrelevant features
#     all_data = all_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
#
#     # Separate the train and test data
#     train_encoded = all_data[:len(train)]
#     test_encoded = all_data[len(train):]
#
#     # Encode categorical features using LabelEncoder
#     categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']
#     for col in categorical_cols:
#         le = LabelEncoder()
#         train_encoded[col] = le.fit_transform(train_encoded[col].fillna('NA'))
#         test_encoded[col] = le.transform(test_encoded[col].fillna('NA'))
#
#     return train_encoded, test_encoded
#
#
# def titanic_prediction(train_file, test_file):
#     # Load the data
#     train = pd.read_csv(train_file)
#     test = pd.read_csv(test_file)
#
#     # Feature engineering
#     train_encoded, test_encoded = feature_engineering(train, test)
#
#     train = train_encoded
#     test = test_encoded
#
#     # Concatenate the training and test data
#     all_data = pd.concat([train, test], ignore_index=True)
#
#     # Encode categorical features
#     le = LabelEncoder()
#     categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']
#
#     # Fit LabelEncoder on all data to ensure all categories are seen
#     for col in categorical_cols:
#         le.fit(all_data[col].fillna('NA'))
#         train[col] = le.transform(train[col].fillna('NA'))
#         test[col] = le.transform(test[col].fillna('NA'))
#
#     # Split the data into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(train.drop('Survived', axis=1), train['Survived'], test_size=0.2, random_state=42)
#
#     # Model selection
#     models = {
#         'Logistic Regression': LogisticRegression(),
#         'Random Forest': RandomForestClassifier(),
#         'Gradient Boosting': GradientBoostingClassifier()
#     }
#
#     for name, model in models.items():
#         print(f'Training {name}...')
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_val)
#         print(f'Accuracy on validation set for {name}: {accuracy_score(y_val, y_pred)}')
#
#     # Hyperparameter tuning
#     rf_params = {
#         'n_estimators': [100, 200, 300, 400, 500],
#         'max_depth': [5, 10, 15, 20],
#         'min_samples_split': [2, 5, 10, 15]
#     }
#
#     rf = RandomForestClassifier(random_state=42)
#     rf_cv = GridSearchCV(rf, rf_params, cv=5, n_jobs=-1)
#     rf_cv.fit(X_train, y_train)
#
#     print(f'Best hyperparameters: {rf_cv.best_params_}')
#     print(f'Accuracy on validation set: {rf_cv.best_score_}')
#
#     # Ensembling
#     models = [
#         RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42),
#         RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=10, random_state=42),
#         GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
#     ]
#
#     predictions = []
#     for model in models:
#         model.fit(X_train, y_train)
#         y_pred = model.predict(test.drop('PassengerId', axis=1))
#         predictions.append(y_pred)
#
#     y_pred = (sum(predictions) >= len(predictions) / 2).astype(int)
#
#     # Save the predictions to a file
#     submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_pred})
#     submission.to_csv('submission.csv', index=False)
#
#     print('Predictions saved to submission.csv')

def build_the_model(train_file, test_file):

    # Dropping unnecessary columns
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    train_df = train_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    test_df = test_df.drop(["Name", "Ticket", "Cabin"], axis=1)

    # Taking care of missing data
    data = [train_df, test_df]

    for dataset in data:
        mean = train_df["Age"].mean()
        std = test_df["Age"].std()
        is_null = dataset["Age"].isnull().sum()
        # compute random numbers between the mean, std and is_null
        rand_age = np.random.randint(mean - std, mean + std, size=is_null)
        # fill NaN values in Age column with random values generated
        age_slice = dataset["Age"].copy()
        age_slice[np.isnan(age_slice)] = rand_age
        dataset["Age"] = age_slice
        dataset["Age"] = train_df["Age"].astype(int)
    train_df["Age"].isnull().sum()

    # 'Embarked' in Train set
    common_value = 'S'
    train_df["Embarked"] = train_df["Embarked"].fillna(common_value)

    # 'Fare' in Test set
    test_df = test_df.fillna(test_df['Fare'].mean())

    # Encoding categorical data
    le = LabelEncoder()
    train_df["Sex"] = le.fit_transform(train_df["Sex"])
    train_df["Embarked"] = le.fit_transform(train_df["Embarked"])

    test_df["Sex"] = le.fit_transform(test_df["Sex"])
    test_df["Embarked"]= le.fit_transform(test_df["Embarked"])

    # Splitting the Train & Test datasets
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test = test_df.drop("PassengerId", axis=1).copy()

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # 2.0 Training Model using Decision Tree
    decisionTree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    decisionTree.fit(X_train, Y_train)
    Y_pred = decisionTree.predict(X_test)
    decisionTree.score(X_train, Y_train)

    # 3. - Creating a submission
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
    submission.to_csv('submission.csv', index=False)
    print('Predictions saved to submission.csv')


if __name__ == '__main__':
    # build_random_forest(train_file='train.csv', test_file='test.csv')
    # titanic_prediction('train.csv', 'test.csv')
    build_the_model(train_file='train.csv', test_file='test.csv')


