from unittest import TestCase
from main import Classifier

class TestClassifier(TestCase):
    def setUp(self):
        pass 

    # Two Label Categories
    def test_one_feature(self):
        feature = {"petal_length": [0.1, 0.2, 0.48, 0.5, 1]}
        label = [0,0,0,1,1]
        
        classifier = Classifier()
        classifier.train(feature=feature, label=label)

    def test_two_feature(self):
        pass 

        