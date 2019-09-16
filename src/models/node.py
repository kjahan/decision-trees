class Node:
    def __init__(self, instances, target_variable):
        self.children = {}
        self.feature_name = None
        self.instances = instances
        self.class_label = None
        self.target = target_variable

    def set_feature_name(self, fet_name):
        """
        Set feature name chosen for current node
        :param fet_name: feature name
        """
        self.feature_name = fet_name

    def add_child(self, fet_value, node):
        """
        Add a child to current node where its feature name has given value
        :param fet_value: feature value for this child
        :param node: child node
        """
        self.children[fet_value] = node

    def assign_label(self, label):
        """
        If this is an external node, assign label to it for prediction
        :param label:
        """
        self.class_label = label

    def get_target_labels(self):
        """
        Get all possible labels for instances assigned to current node
        :return:
        """
        return list(set(self.instances[self.target].values))

    def is_pure(self):
        """
        Check if current node is pure
        :return: True if current node is pure otherwise false
        """
        if len(self.get_target_labels()) == 1:
            return True
        return False

    def get_label(self):
        """
        Get class label from current node if its external/pure node
        :return:
        """
        if self.is_pure():
            return self.class_label
        else:
            raise ValueError
