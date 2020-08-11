"""
Implementation of a Naive Bayes Classifier for numeric features.
"""

import math
from typing import Tuple
import pandas as pd
from scipy.stats import norm

class NaiveBayesClassifier:
    """Naive Bayes Classifier implementation.

        Uses a normal distribution to model each class-conditional distribution.
    """

    def __init__(self):
        self.classes = set()
        self.probabilities = {}
        self.parameters = {}

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame) -> None:
        """Fits a Naive Bayes Classifier to the provided training data.

        All features must be numeric.

        Args:
            features: A pd.DataFrame of numeric feature values.
            labels: A pd.Series of classification labels.
        """
        # Find all classes in the data, and the overall probability of each.
        self.classes = set(labels)
        for label_class in self.classes:
            self.probabilities[label_class] = (
                labels.value_counts()[label_class] / len(labels))

        # For each feature, use maximum likelihood estimation to fit a normal
        # distribution, and store the mean and standard deviation.
        for feature in features:
            self.parameters[feature] = {}
            for label_class in self.classes:
                class_idx = labels.index[labels == label_class]
                values_from_class = features[feature][class_idx]
                mu, sigma = norm.fit(values_from_class)
                self.parameters[feature][label_class] = (mu, sigma)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predicts labels for given test features.

        Args:
            features: A pd.DataFrame of numeric feature values.

        Returns:
            A pd.Series of predicted labels.
        """
        def calc_log_probs(row):
            class_probs = {}
            for label_class in self.classes:
                class_prob = 0
                for feature in features.columns:
                    mu, sigma = self.parameters[feature][label_class]
                    class_prob += math.log(norm.pdf(row[feature], mu, sigma))
                class_prob += math.log(self.probabilities[label_class])
                class_probs[label_class] = class_prob
            return class_probs

        calculated_probs = list(features.apply(calc_log_probs, axis=1))
        predictions = pd.Series([max(d, key=d.get) for d in calculated_probs],
                                index=features.index)
        return predictions


def train_test_split(
        data: pd.DataFrame, random_state: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a dataframe 80/20 into a train and test set.

    Args:
        data: A pd.DataFrame to split
        random_state: An optional random seed (for repeatability)
    """
    if random_state is None:
        train = data.sample(frac=0.8)
    else:
        train = data.sample(frac=0.8, random_state=random_state)
    test = data.drop(train.index)
    return (train, test)

def score(predicted: pd.Series, actual: pd.Series) -> float:
    """Computes the proportion of matches between predicted and actual labels.

    Args:
        predicted: A pd.Series of predicted categorical labels
        actual: A pd.Series of actual categorical labels

    Returns:
        The proportion of matching labels
    """
    return sum(predicted == actual) / len(predicted)
