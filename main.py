from typing import Optional

class Node:
    def __init__(self, features:str, value: int | float):
        self.features = features
        self.value = value
        
        self.left: Optional[Node] = None # less than
        self.right: Optional[Node] = None # greater and equal

class Classifier:
    def __init__(self):
        self.accuracy_threshold = 0.6

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
 

    def _validate_training_data(self, features: dict[str, list[int | float]], label: list[int]):
        label_count = len(label)

        for feat, vals in features.items():
            val_count = len(vals)
            if val_count != label_count:
                raise ValueError(f"Feature {feat} values count is not equal to label count.")
            
    def train(self, features: dict[str, list[int | float]], label: list[int]):
        self._validate_training_data(features, label)

    def predict(self, features: list[float]):
        pass 




