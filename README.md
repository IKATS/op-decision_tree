# Operator op-decision_tree

## Decision Tree fit

### Input and parameters

This operator only takes one input of the functional type **table**.

It also takes 4 inputs from the user :

- **Target** : the name of the variable we want to predict in the input table
- **ID** : the name of the rows (or table key) of the input table
- **Max Depth** : the maximum depth of the tree. The default value of zero means there is no constraint on the depth of the tree.
- **Class Balancing** : Apply weightbalancing on the items inversely proportional to class frequencies in the input data

**Class Balancing** is optional, default False: when True: apply a weight balancing on the classes,inversely proportional to class frequencies in the input data, according to the following formula:

weight(label) = total_samples / (nb_classes\*count_samples(label))

### Outputs

The operator has three outputs :

- **TDT** : a special format used by TDT viztool to show details about the built model
- **Model** : a binary dump of the best model found by the procedure, to be used by the Decision Tree Predict operator
- **Dot** : the visualisation of the best found decision tree in the GraphViz format

## Decision Tree Predict

This IKATS operator implements predict algorithm for DecisionTree of `scikit-learn`

### Input and parameters

This operator takes two inputs :

- **Model** : previously trained in [Decision Tree fit](#DecisionTreefit) step
- **Population** : of the functional type **table** (Ex : `test` output from [TrainTestSplit](https://ikats.org/doc/operators/trainTestSplit.html))

It also takes 3 inputs from the user :

- **Target** : the name of the variable we want to predict in the input table
- **ID** : the name of the rows (or table key) of the input table
- **Table name** : output with features and predictions

### Outputs

The operator has two outputs :

- **Confusion** : confusion_matrix as calculated in `scikit-learn`
- **Score** : accuracy score (ratio of correctly predicted observations to the total observations)
