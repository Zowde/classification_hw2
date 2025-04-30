import numpy as np
import pandas as pd
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
    gini = 1.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    if len(data) == 0:
        return 0.0

    # last column
    labels = data[:, -1]
    #count occurrences of each label - coubrs contain the occurency of each class
    ul, counts = np.unique(labels, return_counts=True)

    total = len(labels)

    for count in counts:
        prob = count / total
        gini -= prob ** 2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # we wiil do the same as gini just using log inside the loop
     # last column
    labels = data[:, -1]
    #count occurrences of each label - coubrs contain the occurency of each class
    ul, counts = np.unique(labels, return_counts=True)

    total = len(labels)

    for count in counts:
        prob = count / total
        entropy -= prob * np.log2(prob)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

class DecisionNode:

    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the data instances associated with the node
        self.terminal = False # True iff node is a leaf
        self.feature = feature # column index of feature/attribute used for splitting the node
        self.pred = self.calc_node_pred() # the class prediction associated with the node
        self.depth = depth # the depth of the node
        self.children = [] # the children of the node (array of DecisionNode objects)
        self.children_values = [] # the value associated with each child for the feature used for splitting the node
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.chi = chi # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func # the impurity function to use for measuring goodness of a split
        self.gain_ratio = gain_ratio # True iff GainRatio is used to score features
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node's prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # extract the labels (last column of self.data)
        labels = self.data[:, -1]
        
        # count the occurrences of each unique class label
        unique_labels, counts = np.unique(labels, return_counts=True)
        
       # label with the highest count
        pred = unique_labels[np.argmax(counts)]
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
        groups = {} # groups[feature_value] = data_subset
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        #calculate the impurity of the total set
        S_entropy = self.impurity_func(self.data)  # use entropy for impurity calculation
    
        # split the data according to the values of the chosen feature
        feature_values = np.unique(self.data[:, feature])  # get all unique values of the feature
        total_size = len(self.data)
    
         # split the dataset into groups based on the values of the feature
        for value in feature_values:
          group = self.data[self.data[:, feature] == value]  #data where the feature equals 'value'
          groups[value] = group #dictionery where each key is value in feature and the value is data that follows the value
        
        # calculate the impurity for each group
          group_size = len(group)
          group_entropy = self.impurity_func(group)  # calculate the entropy of the group
        
        # calculate the weighted average entropy for the split
          weight = group_size / total_size  # weight based on the size of the group
          goodness += weight * group_entropy  # sum the weighted entropy of the groups
    
    # calculate the goodness of the split as the reduction in impurity
        goodness = S_entropy - goodness  # The difference in entropy
    
        #now we should calculate the entropy to each sub data according specifc value
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
        n_node_sample = len(self.data)
    
        if self.feature != -1:
          gain, _ = self.goodness_of_split(self.feature)
          self.feature_importance = gain * (n_node_sample / n_total_sample)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def split(self):
     """
      Splits the current node according to the self.impurity_func. This function finds
      the best feature to split according to and create the corresponding children.
       This function should support pruning according to self.chi and self.max_depth.

       This function has no return value
    """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
         # Stop if max depth reached
     if self.depth >= self.max_depth:
        self.terminal = True
        return

    # stop if node is pure
     labels = self.data[:, -1]
     if len(np.unique(labels)) == 1:
        self.terminal = True
        return
     best_goodness = -1
     best_feature = None
     best_groups = {}

     n_features = self.data.shape[1] - 1  # exclude label column
 
     for feature in range(n_features):
         goodness, groups = self.goodness_of_split(feature)
 
         # if gain ratio is enabled, calculate it
         if self.gain_ratio:
             split_info = 0
             total_size = len(self.data)
             for group in groups.values():
                 p = len(group) / total_size
                 if p > 0:
                     split_info -= p * np.log2(p)
             if split_info != 0:
                 goodness = goodness / split_info
             else:
                 goodness = 0
 
         if goodness > best_goodness:
             best_goodness = goodness
             best_feature = feature
             best_groups = groups
 
     # if no good split was found, stop splitting
     if best_goodness <= 0 or best_feature is None:
         self.terminal = True
         return
 
     self.feature = best_feature
     self.feature = best_feature

     if self.chi < 1:
        if not self._chi_pass(best_groups):
            self.terminal = True
            return
 
     for value, group_data in best_groups.items():
         child_node = DecisionNode(
             data=group_data,
             impurity_func=self.impurity_func,
             feature=-1,
             depth=self.depth + 1,
             chi=self.chi,
             max_depth=self.max_depth,
             gain_ratio=self.gain_ratio,
        ) 
         self.add_child(child_node, value)
        
         ###########################################################################
        #                              END OF YOUR CODE                            #
        ## #########################################################################                  
class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the training data used to construct the tree
        self.root = None # the root node of the tree
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.chi = chi # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.gain_ratio = gain_ratio #
        
    def depth(self):
        return self.root.depth

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.root = DecisionNode(data=self.data, impurity_func=self.impurity_func, max_depth=self.max_depth, chi=self.chi, gain_ratio=self.gain_ratio)
        self._build_tree(self.root)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    def _build_tree(self, node):
        """
        Recursively builds the decision tree by splitting nodes based on impurity.
        This function will call the split method of the DecisionNode.
        """
        # stop splitting if max_depth is reached 
        if node.depth >= self.max_depth or node.terminal:
            return

        # call the split function to find the best feature to split on
        node.split()

        # recursively build the tree for each child node
        for child in node.children:
            self._build_tree(child)

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        node = self.root
        # traverse the tree based on the instance's feature values
        while not node.terminal:
         feature_value = instance[node.feature]
         if not np.any(node.children_values == feature_value):
             return node.calc_node_pred()  # fallback if unseen value
         index = np.where(node.children_values == feature_value)[0][0]
         node = node.children[index]

         ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        pred = node.pred
        return pred
    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        correct = 0
        for row in dataset:
          sample = row[:-1]  # all features
          true_label = row[-1]  # actual class label
          predicted_label = self.predict(sample)
          if predicted_label == true_label:
               correct += 1
        accuracy = (correct / len(dataset)) * 100
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy
        

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1,2,3,4,5,6,7,8,9,10]:
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Create a decision tree instance for each depth
        tree = DecisionTree(
            data=X_train, 
            impurity_func=calc_entropy, 
            max_depth=max_depth, 
            gain_ratio=False
        )
        
        # Build the tree (sets root and recursively splits)
        tree.build_tree()
        
        # Calculate training accuracy
        train_accuracy = tree.calc_accuracy(X_train)
        training.append(train_accuracy)
        
        # Calculate validation accuracy
        val_accuracy = tree.calc_accuracy(X_validation)
        validation.append(val_accuracy)
        
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes




