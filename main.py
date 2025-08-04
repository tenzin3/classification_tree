class Classifier:
    def __init__(self):
        pass 

    @property
    def training_res(self):
        pass 

    def train(self, feature: dict[str, list[int | float]], label: list[int]):
        # data validation
        label_count = len(label)

        for feat, vals in feature.items():
            val_count = len(vals)
            if val_count != label_count:
                raise ValueError(f"Feature {feat} values count is not equal to label count.")
            
            
        pass 

    def predict(self, feature: list[float]):
        pass 




