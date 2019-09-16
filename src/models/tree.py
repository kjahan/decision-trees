from src.models.node import Node
from src.utilities import get_best_feature

__DEBUG__ = False


class Tree:
    def __init__(self, instances, target_variable):
        self.root = Node(instances, target_variable)
        self.target = target_variable

    def train(self):
        """
        Train ID3 decision trees
        """
        self.construct_tree(self.root)

    def construct_tree(self, node):
        """
        Construct the decision trees using ID3 algorithm
        :param node: current node
        :return:
        """
        target_labels = node.get_target_labels()
        if __DEBUG__:
            print("target labels: {}".format(target_labels))
        if len(target_labels) == 1:
            node.assign_label(target_labels[0])
            if __DEBUG__:
                print("Creating an external pure node!")
            return
        else:
            instances = node.instances
            best_feature_name = get_best_feature(instances, self.target)
            node.set_feature_name(best_feature_name)
            if __DEBUG__:
                print("setting current node fet name: {}".format(node.feature_name))
            for attr_value in instances[best_feature_name].unique():
                if __DEBUG__:
                    print("attr value: {}".format(attr_value))
                child_instances = instances.loc[instances[best_feature_name] == attr_value]
                child_node = Node(child_instances, self.target)
                if __DEBUG__:
                    print("creating a new child node - adding it to parent node: {}".format(node.feature_name))
                node.add_child(attr_value, child_node)
                self.construct_tree(child_node)

    def predict(self, test_instance):
        """
        Given test instance, predict clas based on trained decision tree
        :param test_instance: test instance described as a dictionary of fet:value
        :return: predicted class
        """
        node = self.root
        while not node.is_pure():
            fet_name = node.feature_name
            fet_value = test_instance[fet_name]
            if __DEBUG__:
                print("fet name: {}, fet value: {}".format(fet_name, fet_value))
            node = node.children[fet_value]
            if __DEBUG__:
                print("next visited node: {}".format(node.feature_name))
        prediction = node.get_label()
        if __DEBUG__:
            print("instance predicted label: {}".format(prediction))
        return prediction
