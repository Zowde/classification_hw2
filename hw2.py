import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    last_col = data[:, -1]
    _, counts = np.unique(last_col, return_counts=True)
    # Gini Impurity calculation: 1 - sum(p_i^2) for each class
    gini = 1 - ((counts / data.shape[0])**2).sum()
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    entropy = 0.0
    last_col = data[:, -1]
    _, counts = np.unique(last_col, return_counts=True)
    ratios = (counts / data.shape[0])
    # Entropy calculation: -sum(p_i * log2(p_i)) for each class
    entropy = -((ratios * np.log2(ratios)).sum())
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    return entropy

class DecisionNode:
    def __init__(self, data, impurity_func, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the data instances associated with the node
        self.terminal = False  # True iff node is a leaf
        self.feature = feature  # column index of feature/attribute used for splitting the node
        self.pred = self.calc_node_pred()  # the class prediction associated with the node
        self.depth = depth  # the depth of the node
        self.children = []  # the children of the node (array of DecisionNode objects)
        self.children_values = []  # the value associated with each child for the feature used for splitting the node
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.chi = chi  # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func  # the impurity function to use for measuring goodness of a split
        self.gain_ratio = gain_ratio  # True iff GainRatio is used to score features
        self.feature_importance = 0  # initialize feature importance to 0
    
    def calc_node_pred(self):
        """
        Calculate the node's prediction.

        Returns:
        - pred: the prediction of the node (most frequent class label).
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pred = None
        labels, counts = np.unique(self.data[:, -1], return_counts=True)
        # The prediction is the class label with the highest count
        pred = labels[np.argmax(counts)]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.children.append(node)
        self.children_values.append(val)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {}  # groups[feature_value] = data_subset
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        labels, counts = np.unique(self.data[:, feature], return_counts=True)
        groups = {
            label: self.data[np.where(self.data[:, feature] == label)[0], :]
            for label in labels
        }
        
        if self.gain_ratio:  # If Gain Ratio is used, calculate the information gain ratio
            sum_counts, split_info = 0, 0
            table_entropy = calc_entropy(self.data)
            for index, label in enumerate(labels):
                label_ratio = (counts[index] / self.data.shape[0])
                sum_counts += label_ratio * calc_entropy(groups[label])
                split_info -= label_ratio * np.log2(label_ratio)
            info_gain = table_entropy - sum_counts
            if split_info == 0: 
                goodness = 0
            else:
                goodness = info_gain / split_info

        else:  # If not using Gain Ratio, calculate the impurity reduction
            sum_counts = 0
            for index, label in enumerate(labels):
                sum_counts += (counts[index] / self.data.shape[0]) * self.impurity_func(groups[label])
            goodness = self.impurity_func(self.data) - sum_counts
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return goodness, groups
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.

        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        gos, _ = self.goodness_of_split(self.feature)
        feature_samples = self.data.shape[0]
        # Feature importance is proportional to the goodness of split and the sample size
        self.feature_importance = (feature_samples / n_total_sample) * gos
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and creates the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
   
        if (calc_gini(self.data) == 0) or (self.depth >= self.max_depth):
            self.terminal = True
            return
        
        max_goodness = 0
        all_features = self.data.shape[1]
        for i in range(all_features-1):  # Iterate through each feature
            cur_gos = self.goodness_of_split(i)
            if (cur_gos[0] > max_goodness):  # Find the feature with the best split
                max_goodness, groups = cur_gos
                self.feature = i

        if (self.chi < 1 and cur_gos[0] > 0):  # Chi-square pruning
            feature_chi = 0
            exp_dict = {}
            exp_class_labels, expected_counts = np.unique(self.data[:, -1], return_counts=True)
            expected_ratios = expected_counts / self.data.shape[0]
            
            for class_label, ratio in zip(exp_class_labels, expected_ratios): 
                exp_dict[class_label] = ratio
            DOF = (len(groups)-1) * (len(exp_class_labels)-1)
            
            for label in groups.keys():
                obs_dict = {}
                obs_class_labels, obserevd_counts = np.unique(groups[label][:, -1], return_counts=True)
                
                for class_label, counts in zip(obs_class_labels, obserevd_counts): 
                    obs_dict[class_label] = counts

                for f_class_label in obs_dict.keys():         
                    feature_chi += ((obs_dict[f_class_label] - groups[label].shape[0] * exp_dict[f_class_label]) ** 2) / (groups[label].shape[0] * exp_dict[f_class_label])

            critical_value = chi_table[DOF][self.chi]
            if (feature_chi < critical_value): 
                self.terminal = True
                return

        if max_goodness == 0:
            self.terminal = True
            return 
        
        for label in groups.keys():
            new_node = DecisionNode(groups[label],
                                    impurity_func=self.impurity_func,
                                    depth=(self.depth+1),
                                    chi=self.chi,
                                    max_depth=self.max_depth,
                                    gain_ratio=self.gain_ratio)
            self.add_child(new_node, label)
    
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # Training dataset for constructing the decision tree
        self.root = None  # Tree's root node (initialized later)
        self.max_depth = max_depth  # Maximum depth allowed for the tree
        self.chi = chi  # Chi-square pruning threshold
        self.impurity_func = impurity_func  # Impurity function to be used (e.g., entropy)
        self.gain_ratio = gain_ratio  # Whether to use gain ratio instead of information gain

    def depth(self):
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        return self.root.depth if self.root else 0

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_max_depth(self):
        """Return the maximum depth of the tree using BFS."""
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
    
        if self.root is None:
            return 0
        max_d = 0
        nodes = [self.root]
        while nodes:
            node = nodes.pop()
            max_d = max(max_d, node.depth)
            nodes.extend(node.children)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return max_d

    def build_tree(self):
        """
        Construct the decision tree using the given impurity function.
        Stop only when nodes are pure or splitting yields no gain.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
    
        self.root = DecisionNode(self.data, self.impurity_func, chi=self.chi,
                                 max_depth=self.max_depth, gain_ratio=self.gain_ratio)
        nodes_to_expand = [self.root]
        while nodes_to_expand:
            current = nodes_to_expand.pop()
            current.split()
            nodes_to_expand.extend(current.children)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, instance):
        """
        Predict the label of a given instance by traversing the tree.
        If an unknown feature value is encountered, stop and return current node prediction.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        current = self.root
        while not current.terminal:
            val = instance[current.feature]
            if val in current.children_values:
                next_index = current.children_values.index(val)
                current = current.children[next_index]
            else:
                # if feature value was not seen in training, break early
                break

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return current.pred

    def calc_accuracy(self, dataset):
        """
        Calculate the prediction accuracy of the tree over a dataset.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
    
        correct = 0
        for row in dataset:
            predicted = self.predict(row)
            actual = row[-1]
            if predicted == actual:
                correct += 1
        return (correct / dataset.shape[0])*100
     
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


def depth_pruning(X_train, X_validation):
    """
    Evaluate training and validation accuracy for increasing max_depth values.
    Uses entropy + gain ratio.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    train_acc = []
    val_acc = []

    for depth_limit in range(1, 11):  # From depth 1 to 10
        tree = DecisionTree(X_train, impurity_func=calc_entropy,
                            gain_ratio=True, max_depth=depth_limit)
        tree.build_tree()
        train_acc.append(tree.calc_accuracy(X_train))
        val_acc.append(tree.calc_accuracy(X_validation))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return train_acc, val_acc


def chi_pruning(X_train, X_validation):
    """
    Evaluate accuracy and depth for different chi-square cutoff values.
    Uses entropy + gain ratio.
    """
    chi_train = []
    chi_val = []
    tree_depths = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    chi_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001, ]
    for chi_value in chi_values:
        tree = DecisionTree(
            X_train, 
            impurity_func=calc_entropy,
            gain_ratio=True, 
            chi=chi_value, 
        )
        tree.build_tree()
        
        accuracy_train = tree.calc_accuracy(X_train)
        chi_train.append(accuracy_train)
        
        accuracy_validation = tree.calc_accuracy(X_validation)
        chi_val.append(accuracy_validation)
        
        max_depth = tree.get_max_depth()
        tree_depths.append(max_depth)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return chi_train, chi_val, tree_depths


def count_nodes(node):
    """
    Count the total number of nodes in a tree starting from the given node.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    total = 0
    to_visit = [node, ]

    while len(to_visit) > 0:
        current = to_visit.pop()
        total += 1
        to_visit.extend(current.children)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return total

