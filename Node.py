class Node:
    
    def __init__(self, attribute=None, is_leaf=False, class_label=None, data=None, parent=None):
        self.parent = parent  # The parent node of the current node
        self.attribute = attribute  # The attribute/question based on which the split is made
        self.data = data  # The subset of the dataset that reaches this node
        self.is_leaf = is_leaf  # Flag to check if the node is a leaf
        self.class_label = class_label  # Class label if it's a leaf node
        self.children = {}  # Dictionary to hold children, keys are attribute values, values are Node instances
        

    def add_child(self, value, child_node):
        """
        Adds a child node to this node.

        Parameters:
        - value: The value of the attribute on which the split is made that leads to this child.
        - child_node: The child Node instance to be added.
        """
        self.children[value] = child_node
