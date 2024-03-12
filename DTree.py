"""
This class will serve as the decision tree class.
It will be used to create the decision tree and make predictions.
"""
from Utils import gain_ratio, gain, chi_squared_test
from Node import Node
import sys
import pandas as pd


class DTree:
    def __init__(self, impurity_measure, target, min_rows):
        """
        :param impurity_measure: Entropy, Gini, or Misclassification Error
        :param target: (string) target column name
        :param min_rows: (int) the minimum number of rows in a subset needed to establish a node
        """
        self.impurity_measure = impurity_measure
        self.min_rows = min_rows
        self.root = None
        self.target = target

    def predict(self, row_data):
        """
        Traverse the tree to make a prediction
        :param row_data: df row to predict
        :return:
        """

        current_node = self.root
        if current_node == None:
            return "Unknown"
        else:
            while not current_node.is_leaf:
                if row_data[current_node.attribute] not in current_node.children:
                    return current_node.data['isFraud'].mode()[0]
                current_node = current_node.children[row_data[current_node.attribute]]

            return current_node.class_label

    def build_tree(self, df, parent_node=None):
        """
        This function will build the decision tree based on the data provided
        it starts with a null node;
        will need to be edited for all cases
        :param df: full dataframe at the current attribute node
        :param parent_node: the parent node of the current node/leaf
        :return: The created node of the tree that is added onto the parent node as a child
        """
        best_attribute = self.find_best_split(self.target, df.columns, df)
        num_rows = df.shape[0]

        chi_square = (chi_squared_test(df, best_attribute, self.target) == False) if best_attribute != None else False

        # create a leaf node
        if num_rows < self.min_rows or len(df[self.target].unique()) == 1 or chi_square:
            leaf_value = df[self.target].mode()[0]  # most common class label
            return Node(is_leaf=True, class_label=leaf_value, parent=parent_node, data=df)

        # if there is no best attribute, create a leaf node
        if best_attribute is None:
            leaf_value = df[self.target].mode()[0]
            return Node(is_leaf=True, class_label=leaf_value, parent=parent_node, data=df)

        # get all the different enums for each data label
        subsets = self.get_split(df, best_attribute)

        current_node = Node(is_leaf=False, attribute=best_attribute, parent=parent_node, data=df)
        if current_node.parent == None:
            self.root = current_node

        # Recursively build subtrees for each subset
        for subset in subsets.keys():
            child_node = self.build_tree(subsets[subset], current_node)
            current_node.add_child(subset, child_node)
        return current_node

    def find_best_split(self, target, attribute_names, df):
        """
        This function will find the best split for the data based on the impurity measure
        :param df: dataframe
        :param target  : (string) target column name
        :param attribute_names  : (list) the columns to test for the best split
        :return: (String) of the best attribute to split on
        """

        # Initialize as the smallest number in python
        best_gain = sys.float_info.min
        best_attribute = None

        for attribute in attribute_names:
            if attribute != target:
                att_gain = gain(df, attribute, target, self.impurity_measure)

                if att_gain > best_gain:
                    best_gain = att_gain
                    best_attribute = attribute
        return best_attribute

    def get_split(self, df, attribute_name):
        """
        This function will split the data based on the attribute and threshold provided
        :param df: (pandas series) dataset at the split
        :param attribute_name: (String) attribute to split on
        :return: the split data (dictionary)
        """
        subsets = {}
        for value in df[attribute_name].unique():
            subsets[value] = df[df[attribute_name] == value]
        return subsets

    def print_tree(self, node=None, depth=0):
        """
        Recursively prints the tree structure, starting from the given node.
        :param node: The current node to print. If None, start from the root.
        :param depth: The current depth in the tree (used for indentation).
        """
        if node is None:
            node = self.root

        if node is not None:
            if node.is_leaf:
                print("\t" * depth + f"Leaf: {node.class_label}")
            else:
                print("\t" * depth + f"Node: {node.attribute}")

            for value, child in node.children.items():
                print("\t" * depth + f"- on {value} ->")
                self.print_tree(child, depth + 1)
