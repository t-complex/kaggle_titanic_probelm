import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

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

    build_the_model(train_file='train.csv', test_file='test.csv')


