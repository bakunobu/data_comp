"""
Best models to compete them
"""

from sklearn.tree import DecisionTreeClassifier




best_tree_clf = DecisionTreeClassifier(criterion='gini',
                                  max_depth=8,
                                  max_features='auto',
                                  max_leaf_nodes=7,
                                  min_samples_leaf=5,
                                  min_samples_split=24,
                                  splitter='best')