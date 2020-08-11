"""
Testing Naive Bayes Classifier implementation on Pima Indians Diabetes dataset
(https://www.kaggle.com/uciml/pima-indians-diabetes-database) and Iris dataset
(https://www.kaggle.com/uciml/iris)
"""
import pandas as pd
from naive_bayes_classifier import train_test_split, score, NaiveBayesClassifier

def test(data, label_col):
    """Test the Naive Bayes Classifier on the data and return the accuracy.

    Args:
        data: A pd.DataFrame including features and labels
        label_col: The name of the column in the dataframe containing the labels

    Returns:
        The proportion of observations from the test set classified correctly,
        averaged over 100 trials, each using a random 80/20 train/test split
    """
    num_iters = 100
    avg_score = 0
    for _ in range(num_iters):
        train_set, test_set = train_test_split(data)
        classy = NaiveBayesClassifier()
        train_features = train_set.loc[:, train_set.columns != label_col]
        train_labels = train_set[label_col]
        classy.fit(train_features, train_labels)

        test_features = test_set.loc[:, test_set.columns != label_col]
        test_labels = test_set[label_col]
        test_predictions = classy.predict(test_features)

        avg_score += score(test_predictions, test_labels)
    avg_score /= num_iters
    return avg_score


def main():
    """Test Naive Bayes Classifier on Pima Indians Diabetes and Iris datasets.
    """
    score_diabetes = test(pd.read_csv("./diabetes.csv"), "Outcome")
    print("Diabetes score:", score_diabetes)
    iris_data = pd.read_csv("./iris.csv")
    iris_data = iris_data.drop("Id", axis=1)
    score_iris = test(iris_data, "Species")
    print("Iris score:", score_iris)


if __name__ == "__main__":
    main()
