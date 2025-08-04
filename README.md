# Classification Tree

A Python implementation of a binary classification decision tree algorithm. This project provides a simple yet effective way to build decision trees for binary classification problems.

## Overview

The classification tree algorithm automatically constructs a decision tree by:
- Finding optimal split points for each feature
- Selecting the best feature and threshold combination based on accuracy
- Recursively building left and right subtrees
- Avoiding consecutive splits on the same feature

## Features

- **Binary Classification**: Supports binary classification problems (labels: 0 and 1)
- **Automatic Feature Selection**: Automatically selects the best feature and threshold for each split
- **Accuracy-based Splitting**: Uses accuracy as the criterion for determining optimal splits
- **Configurable Accuracy Threshold**: Set minimum accuracy threshold for node creation
- **Tree Visualization**: Utility function to collect and inspect tree nodes

## Project Structure

```
classification_tree/
├── main.py          # Core implementation of the classification tree
├── test_main.py     # Unit tests and usage examples
└── README.md        # This documentation
```

## Core Classes

### Node
Represents a decision tree node with:
- `feature`: The feature name used for splitting
- `threshold`: The threshold value for the split
- `left_label`: The label assigned to the left branch
- `left`: Left child node (values < threshold)
- `right`: Right child node (values >= threshold)

### Classifier
Main classifier class with methods:
- `train()`: Builds the decision tree from training data
- `predict()`: Makes predictions on new data (to be implemented)
- `calculate_accuracy()`: Calculates prediction accuracy
- Various helper methods for tree construction

## Usage

### Basic Example

```python
from main import Classifier

# Prepare your data
features = {
    "petal_length": [0.1, 0.2, 0.48, 0.5, 1.0],
    "petal_width": [0.9, 0.8, 0.7, 0.8, 0.4]
}
labels = [0, 0, 0, 1, 1]

# Create and train the classifier
classifier = Classifier()
classifier.train(feats=features, labels=labels)

# The tree is now built and ready for predictions
print(f"Root feature: {classifier.root_node.feature}")
print(f"Root threshold: {classifier.root_node.threshold}")
```

### Multi-feature Example

```python
features = {
    "petal_length": [0.2, 0.3, 0.7, 0.8, 0.15, 0.25, 0.9],
    "petal_width": [0.9, 0.8, 0.7, 0.8, 0.4, 0.35, 0.2]
}
labels = [0, 0, 1, 1, 1, 1, 0]

classifier = Classifier()
classifier.train(feats=features, labels=labels)

# Inspect the tree structure
from main import collect_nodes
nodes = collect_nodes(classifier.root_node)
for node in nodes:
    print(f"Feature: {node['feature']}, Threshold: {node['threshold']}, Left Label: {node['left_label']}")
```

## Data Format

### Input Requirements

- **Features**: Dictionary where keys are feature names and values are lists of numerical values
- **Labels**: List of binary labels (0 or 1)
- **Data Consistency**: All feature lists must have the same length as the labels list

### Example Data Structure

```python
features = {
    "feature1": [1.2, 2.3, 0.8, 1.5, 2.1],
    "feature2": [0.5, 0.7, 0.3, 0.9, 0.6],
    "feature3": [10, 15, 8, 12, 14]
}
labels = [0, 1, 0, 1, 1]
```

## Algorithm Details

### Tree Construction Process

1. **Feature Evaluation**: For each feature, the algorithm:
   - Sorts the feature values and corresponding labels
   - Tests all possible split points
   - Evaluates both possible label assignments (0/1 and 1/0 for left/right)
   - Selects the split with highest accuracy

2. **Root Selection**: The feature with the highest accuracy becomes the root node

3. **Recursive Splitting**: 
   - Data is split based on the selected threshold
   - Left subtree: values < threshold
   - Right subtree: values >= threshold
   - Process repeats for each subtree

4. **Stopping Criteria**:
   - No feature achieves accuracy above the threshold (default: 0.6)
   - Same feature cannot be used for consecutive splits
   - No more data to split

### Configuration

You can modify the accuracy threshold when creating a classifier:

```python
classifier = Classifier()
classifier.accuracy_threshold = 0.8  # Higher threshold = fewer splits
```

## Testing

Run the included tests to verify functionality:

```bash
python -m pytest test_main.py
```

The tests include:
- Single feature classification
- Multi-feature classification with complex tree structures
- Verification of tree node properties

## Limitations

- Currently supports only binary classification (0/1 labels)
- The `predict()` method is not yet implemented
- No pruning or regularization techniques
- Assumes numerical features only

## Future Enhancements

- Implement the `predict()` method for making predictions
- Add support for multi-class classification
- Implement tree pruning techniques
- Add support for categorical features
- Include visualization tools for the decision tree

## Requirements

- Python 3.7+
- No external dependencies required

## License

This project is open source and available under the MIT License.