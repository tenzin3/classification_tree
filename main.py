from typing import Optional

class Node:
    def __init__(self, feature:str, value: int | float):
        self.feature = feature
        self.value = value
        
        self.left: Optional[Node] = None # less than
        self.right: Optional[Node] = None # greater and equal

class Classifier:
    def __init__(self):
        self.accuracy_threshold = 0.6

    @property
    def training_res(self):
        pass 

    def _validate_training_data(self, feature: dict[str, list[int | float]], label: list[int]):
        label_count = len(label)

        for feat, vals in feature.items():
            val_count = len(vals)
            if val_count != label_count:
                raise ValueError(f"Feature {feat} values count is not equal to label count.")
            
        
    def calculate_accuracy(self, prediction: list[int], label: list[int]):
        if len(prediction) != len(label):
            raise ValueError("Number of prediction values is not equal to Labels.")
        
        correct = 0
        for pred, lab in zip(prediction, label):
            if pred == lab:
                correct += 1
        
        return correct / len(prediction)

    def train(self, feature: dict[str, list[int | float]], label: list[int]):
        self._validate_training_data(feature, label)

    def predict(self, feature: list[float]):
        pass 




