from typing import Optional

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

    @property
    def training_res(self):
        pass

    @staticmethod
    def calculate_accuracy(prediction: list[int], label: list[int]):
        if len(prediction) != len(label):
            raise ValueError("Number of prediction values is not equal to Labels.")
        
        correct = 0
        for pred, lab in zip(prediction, label):
            if pred == lab:
                correct += 1
        
        return correct / len(prediction)
    
    @staticmethod
    def get_threshold_index(arr: list[int | float], threshold: int | float) -> int:
        for idx, val in enumerate(arr):
            if val >= threshold:
                return idx
 
    def _validate_training_data(self, features: dict[str, list[int | float]], label: list[int]):
        label_count = len(label)

        for feat, vals in features.items():
            val_count = len(vals)
            if val_count != label_count:
                raise ValueError(f"Feature {feat} values count is not equal to label count.")
            
    def _sort_feature_values(self, feature_vals: list[int | float], label: list[int]):
        # Get the sorted indices based on vals
        sorted_indices = sorted(range(len(feature_vals)), key=lambda i: feature_vals[i])

        # Sort both lists using the indices
        sorted_vals = [feature_vals[i] for i in sorted_indices]
        sorted_label = [label[i] for i in sorted_indices]

        return sorted_vals, sorted_label
    
    from typing import Union

    def _sort_features_and_label(self,
                                features: dict[str, list[Union[int, float]]],
                                label: list[int],
                                feature_name: str
        ) -> tuple[dict[str, list[Union[int, float]]], list[int]]:
        # Get the list of values for the given feature
        feature_values = features[feature_name]

        # Obtain indices that would sort the feature_values
        sorted_indices = sorted(range(len(feature_values)), key=lambda i: feature_values[i])

        # Reorder each feature list using sorted indices
        sorted_features = {
            fname: [features[fname][i] for i in sorted_indices]
            for fname in features
        }

        # Reorder the label list using sorted indices
        sorted_label = [label[i] for i in sorted_indices]

        return sorted_features, sorted_label


    def _walk(self, node: Node, features: dict[str, list[int | float]], label: list[int]):
        # first walk
        if node == None:
            max = 0
            root_node = None
            for feat, vals in features.items():
                sorted_vals, sorted_lab = self._sort_feature_values(vals, label)

                feat_max = 0
                length = len(sorted_vals)
                for i in range(length):
                    # Left 0 and Right 1
                    pred = [0] * (i-0) + [1] * (length - i) 
                    acc = self.calculate_accuracy(pred, sorted_lab)
                    if acc > feat_max:
                        feat_max = acc
                        root_node = Node(
                            feature=feat, threshold=sorted_vals[i], label=0
                        )
                        
                    # Left 1 and Right 0
                    pred = [1] * (i-0) + [0] * (length - i)
                    acc = self.calculate_accuracy(pred, sorted_lab)
                    if acc > feat_max:
                        feat_max = acc
                        root_node = Node(
                            feature=feat, threshold=sorted_vals[i], label=1
                        )
                
                if feat_max > max:
                    max = feat_max

            self.root_node = root_node
            sorted_features, sorted_label = self._sort_features_and_label(features, label, self.root_node.feature)

            threshold_index = self.get_threshold_index(sorted_features[self.root_node.feature], self.root_node.threshold)
            # left side
            sliced_features = {key: value[:k] for key, value in features.items()}
            self._walk(self.root_node, sliced_features, sorted_label[:threshold_index])
            # right side
            sliced_features = {key: value[k:] for key, value in features.items()}
            self._walk(self.root_node, sliced_features, sorted_label[k:])
            
    def train(self, features: dict[str, list[int | float]], label: list[int]):
        self._validate_training_data(features, label)

        root_node = None
        self._walk(root_node, features, label)

    def predict(self, features: list[float]):
        pass 




