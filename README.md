## [Lab 1. Data preparation for classification task](https://github.com/sogoeslight/AI_Labs/tree/master/src/main/scala/com/sogoeslight/task/preparation)
### Task:
1) Choose dataset
2) Load dataset and prepare it as a data structure
3) Output info about data structure size (amount of samples, attributes, ..)
4) Remove rows, containing empty values
5) Convert categorical attributes to the binary ones
6) Output first 5 rows
7) Divide dataset on 2 samples (75/25), 75 - learning samples, 25 - testing samples


## [Lab 2. Decision tree and its accuracy](https://github.com/sogoeslight/AI_Labs/blob/master/src/main/scala/com/sogoeslight/task/tree/)
### Task:
1) Prepare the dataset and split it into test and learning samples
2) Implement algorithm for generating decision tree 
(recommended one is "Iterative Dichotomiser")
3) Train algorithm
4) Output amount of leaves, depth, metrics
5) Calculate and output tree accuracy
6) Print tree


## [Lab 3. Logistic Regression](https://github.com/sogoeslight/AI_Labs/tree/master/src/main/scala/com/sogoeslight/task/regression)
### Task:
1) Prepare the dataset and split it into test and learning samples
2) Implement algorithm for logistic regression model training
3) Train algorithm
4) Output model properties (coefficients, free term)
5) Calculate and output regression accuracy


## [Lab 4. One Layer Neural Network](https://github.com/sogoeslight/AI_Labs/tree/master/src/main/scala/com/sogoeslight/task/regression)
### Task:
1) Prepare the dataset and split it into test and learning samples
2) Implement algorithm for one layer perceptron training
3) Train algorithm
4) Output model properties (vector of coefficients weights, free term)
5) Calculate and output regression accuracy
6) Investigate how `max_iter` value influences model performance. Display it.


## [Lab 5. Multilayer Neural Network for classification tasks](https://github.com/sogoeslight/AI_Labs/tree/master/src/main/scala/com/sogoeslight/task/regression/MultiLayerPerceptrone.scala)
### Task:
1) Prepare the dataset and split it into test and learning samples
2) Implement algorithm for multi layer perceptron training
3) Train algorithm
4) Output model properties (vector of coefficients weights, free term)
5) Calculate and output regression accuracy
6) Investigate how `max_iter` value influences model performance
7) Investigate how amount of hidden layers and amount of neurons in them impact performance
   

## Lab 6. [Multilayer Neural Network](https://github.com/sogoeslight/AI_Labs/tree/master/src/main/scala/com/sogoeslight/task/regression/MultiLayerRegression.scala) and [Regression Tree](https://github.com/sogoeslight/AI_Labs/tree/master/src/main/scala/com/sogoeslight/task/tree/RegressionTree.scala) for Regression Tasks
### Task:
1) [Prepare the dataset](https://github.com/sogoeslight/AI_Labs/tree/master/src/main/scala/com/sogoeslight/task/preparation/Data_Akbilgic.scala) “ISTANBUL STOCK EXCHANGE”
   https://archive.ics.uci.edu/ml/datasets/ISTANBUL+STOCK+EXCHANGE
2) Dataset contains observations of 9 stock indexes. Choose any of them as a dependent variable, other are going to be used as independent ones
3) Calculate Pearson's correlation coefficients for each possible pair of variables.
4) Implement algorithm for multi layer perceptron to solve regression task
5) Implement algorithm for regression tree to solve regression task
6) Train algorithms and calculate and output regression accuracy
7) Output model's parameters. Weights and free term for MLNN and tree depth and amount of leaves for the Regression Tree
8) Compare models