"""
This file will contain utility functions that will be used in the decision tree algorithm
Intentionally created only as a set of functions and not a class
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import math


def gini_index(attribute_series):
    """
    This function will take an attribute/feature series and calculate the gini index of the column
    and calculates the probability of each class in the column

    :param attribute_series: feature being tested (SERIES from pandas)
    :return: gini index
    """
    gini = 1
    probabilities = calculate_probability(attribute_series)
    for prob in probabilities:
        gini -= prob ** 2

    return gini


def misclassification_error(attribute_series):
    """
    This function will take an attribute/feature series and calculate the misclassification error of the column
    :param attribute_series: feature being tested (SERIES from pandas)
    :return: 1 - max(probabilities)
    """
    mc_error = 1
    probabilities = calculate_probability(attribute_series)
    return mc_error - max(probabilities)


def entropy(attribute_series):
    """
    This function will take an attribute/feature series and calculate the entropy of the column
    :param attribute_series: feature being tested (SERIES from pandas)
    :return: entropy
    """
    entropy_att = 0
    probabilities = calculate_probability(attribute_series)
    probabilities = list(filter(lambda a: a != 0, probabilities))

    for prob in probabilities:
        entropy_att -= prob * np.log2(prob)

    return entropy_att


def gain(init_df, attribute, target, imp_measure):
    """
    :param init_df: full dataframe 
    :param attribute: STRING name of the attribute/feature being tested
    :param target: STRING target of the target column
    :param imp_measure: function to calculate impurity
    :return: information gain
    """

    # Calculate the overall impurity of the original data
    info_gain = imp_measure(init_df[target])

    # Split the data based on the attribute values
    attribute_series = init_df[attribute]
    vals_count_dict = attribute_series.value_counts().to_dict()  # counts for each value in the attribute column

    for (val, count) in vals_count_dict.items():
        # Calculate the impurity of the split
        split_impurity = imp_measure(
            init_df[init_df[attribute] == val][target])  # impurity of each split (based on value of attribute)
        info_gain -= (count / len(init_df)) * split_impurity

    return info_gain


def gain_ratio(init_df, attribute, target, imp_measure):
    """
    :param init_df: full dataframe
    :param attribute: STRING name of the attribute/feature being tested
    :param target: STRING target of the target column
    :param imp_measure: function to calculate impurity
    :return: gain ratio
    """

    info_gain = gain(init_df, attribute, target, imp_measure)
    unique_vals = len(init_df[attribute].unique())
    if unique_vals == 1:  # to avoid division by zero
        return 0
    else:
        split_info = 0
        for p in (init_df[attribute].value_counts(normalize=True)):
            split_info -= p * math.log2(p)

    print(f"{attribute} info_gain: {info_gain}, split_info: {split_info}")
    return info_gain / split_info


def calculate_probability(attribute_df):
    """
    This function will take an attribute/feature dataframe series and
    calculate the probability of each value in the column using a built-in Pandas function
    called value_counts().
    :return: A list of the probabilities of each value in the column
    """
    att_val_probs = attribute_df.value_counts(normalize=True)
    return att_val_probs.values.tolist()  # return the probabilities as a list


def chi_squared_test(df, attribute, target):
    """
    This function will perform the Chi-squared test to determine if the attribute is relevant to the target variable.
    :param df: dataframe
    :param attribute: (string) attribute
    :param target: (string) target
    :return: boolean if the attribute is relevant to the target variable or not
    """
    # Create a contingency table
    contingency_table = pd.crosstab(df[attribute], df[target])
    # Perform the Chi-squared test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    # Decide if the attribute is relevant based on the p-value
    alpha = 0.5  # significance level
    if p < alpha:
        return True
    else:
        return False


def bin_series(series, num_bins=2):
    """
    This function will take a series and bin it into a specified number of bins
    :param num_bins: number of bins we want to place the continuous data in (equal distribution)
    :param series: (pandas series) the series to be binned
    :return: (pandas series) the binned series
    """
    if pd.api.types.is_numeric_dtype(series):
        bin_cat = np.arange(1, num_bins + 1)
        bins = pd.cut(series, num_bins, labels=bin_cat)
        return bins
    return series


def bin_all_columns(df, num_bins=2):
    """
    This function will take a dataframe and bin all the columns
    :param num_bins: number of bins we want to place the continuous data in
    :param df: (pandas dataframe) the dataframe to be binned
    :return: (pandas dataframe) the binned dataframe
    """
    for col in df.columns:
        df[col] = bin_series(df[col], num_bins)
    return df


def undersample(X_train, y_train):
    """
    This function will take the training data and perform undersampling to balance the target classes
    since the notFraud class is much larger than the Fraud class
    :param X_train: df of features for training
    :param y_train: df of target for training
    :return: undersampled X_train, y_train
    """
    # concatenate our training data back together
    training_data = pd.concat([X_train, y_train], axis=1)

    # separate minority and majority classes
    not_fraud = training_data[training_data['isFraud'] == 0]
    fraud = training_data[training_data['isFraud'] == 1]

    # downsample majority
    not_fraud_downsampled = not_fraud.sample(n=len(fraud), random_state=42)

    # combine minority and downsampled majority
    downsampled = pd.concat([not_fraud_downsampled, fraud])

    # shuffle the data
    downsampled = downsampled.sample(frac=1, random_state=42)

    return downsampled.drop(columns=['isFraud']), downsampled['isFraud']


def calc_fraud_rate(df):
    """
    Calculate the weighted fraud rate for each unique card in the dataset.
    
    :param data: DataFrame containing 'card1' and 'isFraud' columns
    :return: Series containing the weighted fraud rate for each card
    """
    fraud_rate = df.groupby('card1')['isFraud'].mean()
    card_frequency = data['card1'].value_counts()
    card_weight = 1 / card_frequency
    total_weight = card_weight.sum()
    card_weight_normalized = card_weight / total_weight

    # Calculate the weighted fraud rate for each card
    weighted_fraud_rate = fraud_rate * card_weight_normalized

    return weighted_fraud_rate
