from unittest import TestCase
from main import Classifier

class TestClassifier(TestCase):
    def setUp(self):
        pass 

    # Two Label Categories
    def test_one_feature(self):
        features = {"petal_length": [0.1, 0.2, 0.48, 0.5, 1]}
        labels = [0,0,0,1,1]
        
        classifier = Classifier()
        classifier.train(features=features, labels=labels)

        assert classifier.root_node.feature == "petal_length"
        assert classifier.root_node.left_label == 0
        assert classifier.root_node.threshold == 0.5
        assert classifier.root_node.left == None
        assert classifier.root_node.right == None

    def test_two_feature(self):
        features = {
            "petal_length": [0.2, 0.3, 0.7, 0.8, 0.15, 0.25, 0.9],
            "petal_width": [0.9, 0.8, 0.7, 0.8, 0.4, 0.35, 0.2]
        }
        labels = [0,0,1,1,1,1,0]
        classifier = Classifier()
        classifier.train(features=features, labels=labels)

        