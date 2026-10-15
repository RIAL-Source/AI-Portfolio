# Mushroom Edibility Prediction with Decision Trees

This project implements a decision tree classifier from scratch in Python to predict whether a mushroom is edible or poisonous based on its features. The model is trained on a small dataset of mushroom samples with binary features.

## Dataset

The training data is a small manually defined dataset (`X_train` and `y_train`) where each sample has three binary features and a target label:
- `X_train`: 10 mushroom samples, each with 3 binary features (shape: 10 x 3)  
- `y_train`: Binary target for each sample (1 = edible, 0 = poisonous)

## Decision Tree Implementation

The decision tree is built using a recursive algorithm that splits the data into left and right subsets at each node based on the feature that gives the maximum information gain. 

The key components are:

1. `compute_entropy(y)`: Calculates the entropy of a node 
2. `split_dataset(X, node_indices, feature)`: Splits the data at a node into left and right branches based on a feature
3. `compute_information_gain(X, y, node_indices, feature)`: Computes the information gain of splitting a node on a feature
4. `get_best_split(X, y, node_indices)`: Finds the best feature to split a node based on maximum information gain
5. `build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth)`: Builds the decision tree recursively up to a specified max depth

The tree is printed out at each node showing the split feature and the sample indices in each branch.

## Usage

Run the `main.py` script to see the decision tree being built and printed:

```bash
python main.py
```

The script will:
1. Compute entropy at the root node
2. Demonstrate splitting the data at a node into left and right branches
3. Find the best feature to split the root node 
4. Build the full decision tree up to `max_depth=2` and print out the tree

## Future Enhancements

- Add a `predict()` function to make predictions on new mushroom samples
- Implement tree pruning to handle overfitting
- Extend to handle non-binary features
- Validate on a larger mushroom dataset

## Acknowledgments

This project was developed as part of a machine learning course on Coursera. The skeleton code and datasets are based on programming assignments from the course.