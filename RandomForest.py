from statistics import mode
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from math import log, ceil

from Utils import gini_index, misclassification_error, entropy, gain, chi_squared_test
import Node
from DTree import DTree

class RandomForest:
    def __init__(self, impurity_measure, target, count=20, min_rows=0):
        """
            :param impurity_measure: entropy, gini, or mis classification error
            :param target: (string) target column name
            :param count: how many trees are being made
            :param min_rows: (int) the minimum number of rows in a subset needed to establish a non-leaf node
            """
        self.trees = []
        self.impurity_measure = impurity_measure
        self.min_rows = min_rows
        self.target = target
        self.count = count

    def bag(self, X_train, y_train):
        """
                performs bagging (bootstrap aggregating). Does "bootstrapping" on the original dataset to make "count"
                    samples, creates "count" decision tree classifiers, and aggregates those individual trees into a single ensemble model
                self: the RandomForest
                df: the data frame to be bootstrapped
                count (int): the number of bootstrapping samples to collect
                output: a single ensemble model
            """
        
        # number of columns and rows to randomly select
        cols = len(X_train.axes[1]) # total # of features = 26 (not including isFraud)
        rows = len(X_train.axes[0]) # total # of rows = 472431 in train, 118108 in test
        X_train = X_train.sample(n=ceil(log(cols)), axis='columns', replace=False)
        sampled_data = X_train.join(y_train)
        
        # randomly select m rows
        sampled_data = sampled_data.sample(n=ceil(log(rows)), replace=False)

        return sampled_data

    def build_forest(self, X_train, y_train):
        """
        utilizes bagging function and the DTree class to create a random forest
        :param X_train: df of features
        :param y_train: series containing only the target column
        :return: n/a
        """
        for _ in range(self.count):
            # Use bagging to create dataset sample
            sample_data = self.bag(X_train, y_train)

            # Decision tree classifier
            decision_tree = DTree(self.impurity_measure, self.target, self.min_rows)
            decision_tree.build_tree(sample_data)
            self.trees.append(decision_tree)

    def aggregate_predictions(self, X):
        """
            Aggregates predictions from each decision tree in forest
            :param forest: list of decision tree classifiers in random forest
            :param X: Dataframe, input features
        """

        row_count = len(X.axes[0])
        all_predictions = []
        for row in range(row_count):
            row_predictions = []
            row_data = X.iloc[row]
            counter = 0
            for tree in self.trees:
                tree_prediction = tree.predict(row_data)
                if tree_prediction == "Unknown":
                    continue
                else:
                    row_predictions.append(tree_prediction)
            
            # Aggregate
            if row_predictions != []:
                all_predictions.append(mode(row_predictions))
        return all_predictions

    def print_forest(self):
        """
        prints the trees in the forest
        """
        counter = 0
        for tree in self.trees:
            print(f"Tree #{counter}")
            tree.print_tree()
            counter = counter + 1
