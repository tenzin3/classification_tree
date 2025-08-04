from typing import Optional

feats_dtype = dict[str, list[int | float]]

class Node:
    def __init__(self, feature:str, threshold: int | float, label: int):
        self.feature = feature
        self.threshold = threshold
        self.left_label = label # always store what label comes to left
        
        self.left: Optional[Node] = None # less than
        self.right: Optional[Node] = None # greater and equal

class Classifier:
    def __init__(self):
        self.accuracy_threshold = 0.6
        self.root_node: Optional[Node] = None

    @staticmethod
    def calculate_accuracy(predictions: list[int], labels: list[int]):
        if len(predictions) != len(labels):
            raise ValueError("Number of prediction values is not equal to Labels.")
        
        correct = 0
        for pred, lab in zip(predictions, labels):
            if pred == lab:
                correct += 1

        return correct / len(predictions)
    
    @staticmethod
    def get_threshold_index(sorted_arr: list[int | float], threshold: int | float) -> int:
        """
        Return index in which sorted array value is greater or equal to threshold.
        """
        for idx, val in enumerate(sorted_arr):
            if val >= threshold:
                return idx
 
    def _validate_training_data(self, features: feats_dtype, labels: list[int]):
        labels_len = len(labels)

        for feat, vals in features.items():
            vals_len = len(vals)
            if vals_len != labels_len:
                raise ValueError(f"Feature {feat} values count is not equal to label count.")
            
    def _sort_feat_and_labels(self, feat: list[int | float], labels: list[int]):
        # Get the sorted indices based on feature values
        sorted_indices = sorted(range(len(feat)), key=lambda i: feat[i])

        # Sort both lists using the indices
        sorted_feat = [feat[i] for i in sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]

        return sorted_feat, sorted_labels

    def _sort_feats_and_labels(self, feats: feats_dtype, labels: list[int], feat_name: str):
        # Get the list of values for the given feature
        feat_vals = feats[feat_name]

        # Obtain indices that would sort the feature_values
        sorted_indices = sorted(range(len(feat_vals)), key=lambda i: feat_vals[i])

        # Reorder each feature list using sorted indices
        sorted_feats = {
            fname: [feats[fname][i] for i in sorted_indices]
            for fname in feats
        }

        # Reorder the label list using sorted indices
        sorted_labels = [labels[i] for i in sorted_indices]

        return sorted_feats, sorted_labels

    def _walk_feat(self, feat: list[int], labels: list[int], feat_name:str):
        sorted_feat, sorted_labs = self._sort_feat_and_labels(feat, labels)

        feat_max = 0
        feat_len = len(sorted_feat)

        node = None
        for i in range(feat_len):
            # Left 0 and Right 1
            pred = [0] * (i-0) + [1] * (feat_len - i) 
            acc = self.calculate_accuracy(pred, sorted_labs)
            if acc > feat_max:
                feat_max = acc
                node = Node(
                    feature=feat_name, threshold=sorted_feat[i], label=0
                )
                
            # Left 1 and Right 0
            pred = [1] * (i-0) + [0] * (feat_len - i)
            acc = self.calculate_accuracy(pred, sorted_labs)
            if acc > feat_max:
                feat_max = acc
                node = Node(
                    feature=feat_name, threshold=sorted_feat[i], label=1
                )
        return feat_max, node


    def _walk(self, node: Node, feats: feats_dtype, labels: list[int]):
        # first walk
        if node == None:
            max = 0
            root_node = None
            for feat_name, feat in feats.items():
                feat_max, node = self._walk_feat(feat, labels, feat_name)
                if feat_max > max:
                    max = feat_max
                    root_node = node

            sorted_feats, sorted_labels = self._sort_feats_and_labels(feats, labels, root_node.feature)

            idx = self.get_threshold_index(sorted_feats[root_node.feature], root_node.threshold)
            # left side
            left_feats = {key: value[:idx] for key, value in feats.items()}
            root_node.left = self._walk(root_node, left_feats, sorted_labels[:idx])
            # right side
            right_feats = {key: value[idx:] for key, value in feats.items()}
            root_node.right = self._walk(root_node, right_feats, sorted_labels[idx:])
            return root_node
        
        # Other than Initial Walk
        max = 0
        next_node = None
        length = len(labels)
        for feat, vals in feats.items():
            # Same feature cant be used for consecutive adjacent nodes
            if feat == node.feature:
                continue

            sorted_vals, sorted_labs = self._sort_feat_and_labels(vals, labels)

            feat_max = 0
            
            for i in range(length):
                # Left 0 and Right 1
                pred = [0] * (i-0) + [1] * (length - i) 
                acc = self.calculate_accuracy(pred, sorted_labs)
                if acc > feat_max:
                    feat_max = acc
                    next_node = Node(
                        feature=feat, threshold=sorted_vals[i], label=0
                    )
                    
                # Left 1 and Right 0
                pred = [1] * (i-0) + [0] * (length - i)
                acc = self.calculate_accuracy(pred, sorted_labs)
                if acc > feat_max:
                    feat_max = acc
                    next_node = Node(
                        feature=feat, threshold=sorted_vals[i], label=1
                    )
            
            # If a particular feature can do greater than accuracy threshold
            if feat_max > max and feat_max > self.accuracy_threshold:
                max = feat_max

        if next_node == None:
            return None
        
        sorted_feats, sorted_labels = self._sort_feats_and_labels(feats, labels, next_node.feature)

        idx = self.get_threshold_index(sorted_feats[next_node.feature], next_node.threshold)
        # left side
        if idx !=0: 
            left_feats = {key: value[:idx] for key, value in feats.items()}
            next_node.left = self._walk(next_node, left_feats, sorted_labels[:idx])
        # right side
        if idx != length - 1:
            right_feats = {key: value[idx:] for key, value in feats.items()}
            next_node.right = self._walk(next_node, right_feats, sorted_labels[idx:])
        return next_node
            
    def train(self, feats: feats_dtype, labels: list[int]):
        self._validate_training_data(feats, labels)

        root_node = None
        self.root_node = self._walk(root_node, feats, labels)

    def predict(self, feat: list[int | float]):
        pass 




