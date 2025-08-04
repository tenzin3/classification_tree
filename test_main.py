from unittest import TestCase
from main import Classifier, collect_nodes

class TestClassifier(TestCase):
    def setUp(self):
        pass 

    # Two Label Categories
    def test_one_feature(self):
        features = {"petal_length": [0.1, 0.2, 0.48, 0.5, 1]}
        labels = [0,0,0,1,1]
        
        classifier = Classifier()
        classifier.train(feats=features, labels=labels)

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
        classifier.train(feats=features, labels=labels)

        nodes = collect_nodes(classifier.root_node)
        expected_nodes = [
            {'feature': 'petal_length', 'threshold': 0.9, 'left_label': 1, 'left': True, 'right': True}, 
            {'feature': 'petal_width', 'threshold': 0.8, 'left_label': 1, 'left': True, 'right': True}, 
            {'feature': 'petal_length', 'threshold': 0.2, 'left_label': 0, 'left': False, 'right': False}, 
            {'feature': 'petal_width', 'threshold': 0.4, 'left_label': 1, 'left': True, 'right': False}, 
            {'feature': 'petal_length', 'threshold': 0.8, 'left_label': 0, 'left': False, 'right': False}, 
            {'feature': 'petal_width', 'threshold': 0.2, 'left_label': 1, 'left': False, 'right': False}
        ]
        assert nodes == expected_nodes
