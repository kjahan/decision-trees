import pandas as pd

from src.models.tree import Tree

__DEBUG__ = False


def load(fn):
    """
    Read csv input file and parse it as Pandas df
    :param fn: filename
    :return: pandas dataframe
    """
    data = pd.read_csv(fn)
    return data


def train_decision_tree(fn):
    """
    Load data in train.csv and train the Decision Tree using ID3 algo
    :param fn: train filename
    :return: trained decision tree
    """
    target = 'Play'
    train_df = load(fn)
    if __DEBUG__:
        print(train_df.head())
    dt = Tree(train_df, target)
    dt.train()
    return dt


def test_decision_tree(fn, dt):
    """
    Load test data and predict class of each test sample
    :param fn: test filename
    :param dt: trained decision tree
    """
    test_df = load(fn)
    if __DEBUG__:
        print(test_df.head())
    samples = test_df.to_dict(orient='records')
    for sample in samples:
        prediction = dt.predict(sample)
        print("Test sample: {} --> predicted class: {}".format(sample, prediction))


if __name__ == "__main__":
    train_fn = "data/train.csv"
    decision_tree = train_decision_tree(train_fn)
    test_fn = "data/test.csv"
    test_decision_tree(test_fn, decision_tree)
