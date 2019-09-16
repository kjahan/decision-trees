import math
import numpy as np

__DEBUG__ = False


def compute_entropy(values):
    """
    Given values for a random variable, calculate its entropy.
    :param values: RV
    :return: entropy of random variable with values outcome
    """
    counts = {}
    for val in values:
        try:
            counts[val] += 1
        except KeyError:
            counts[val] = 1
    entropy = 0
    for key, value in counts.items():
        prob = counts[key] / len(values)
        entropy += -1 * prob * math.log(prob, 2)
    return entropy


def compute_expected_info(data, feature, target):
    """
    Compute expected info for a feature given the observation data
    :param data: training data
    :param feature: feature name to process
    :param target: target variable
    :return: expected info
    """
    data_size = data.shape[0]
    expected_info = 0
    for attr_value in data[feature].unique():
        values = data.loc[data[feature] == attr_value][target].values
        expected_info += compute_entropy(values) * len(values) / data_size
    return expected_info


def get_best_feature(data, target):
    """
    Get best feature that gives the purest node!
    :param data: training data
    :param target: target variable
    :return: feature name
    """
    features = list(set(data.columns) - set([target]))
    exp_info_values = []
    for feature in features:
        exp_info = compute_expected_info(data, feature, target)
        exp_info_values.append(exp_info)
        if __DEBUG__:
            print("feature: {} --> info: {}".format(feature, exp_info))
    best_fet_inx = np.argmin(np.array(exp_info_values))
    if __DEBUG__:
        print("feature: {} --> info: {}".format(features[best_fet_inx], min(exp_info_values)))
    return features[best_fet_inx]
