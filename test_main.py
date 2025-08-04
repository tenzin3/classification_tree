from unittest import TestCase
from main import Classifier

class TestClassifier(TestCase):
    def setUp(self):
        pass 

    # Two Label Categories
    def test_one_feature(self):
        features = {"petal_length": [0.1, 0.2, 0.48, 0.5, 1]}
        label = [0,0,0,1,1]
        
        classifier = Classifier()
        classifier.train(features=features, label=label)

        assert classifier.root_node.feature == "petal_length"
        assert classifier.root_node.left_label == 0
        assert classifier.root_node.threshold == 0.5
        assert classifier.root_node.left == None
        assert classifier.root_node.right == None

    def test_two_feature(self):
        pass 

        