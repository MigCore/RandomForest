# CS 429/529 Project 1 - Random Forest
This README will serve as a guide for the project. It will highlight each file of the project, its uses, and 
provide a guide on how to run the project and how to use it. It will highlight the contributions of each team member.
Lastly, It will provide a summary of the kaggle run. 

## File Manifest:
**main.py** - This file is the main file of the project. It is the file that will be run to execute the project.\
**Node.py** - This file contains the Node class. This class is used to create the nodes of the decision tree.\
**DTree.py** - This file contains the DTree class. This class is used to create the decision tree.\
**RandomForest.py** - This file contains the RandomForest class. This class is used to create the random forest.\
**Utils.py** - This file contains the utility functions used in the project like Information Gain, Gini Index, etc.
**Dcleanin.ipynb** - This file contains the data cleaning process.\

## How to Run:
To run the project, you will have to clean the data first using the Dcleaning.ipynb file. 
To do this, replace the path in cell 2 with the path to the training file
'df = pd.read_csv('resources/train.csv')'
Then replace the path in cell 24 with the path to the training file
'test = pd.read_csv('resources/test.csv')'
There should be two output files that will be used in main and all you will need to do is run the main.py file and will need to have the following libraries installed:\
`pandas`\
`numpy`\
`scipy`\
`sklearn`

## Team Members & Contributions:
### Miguel Cordova 
Collaborated of the DTree class by implementing build_tree and print_tree\
Created Node class\
Cleaned the data to achieve the best results\
### Vincent Hilario
Collaborated on the DTree class by implementing functions like get_split, find_best_split, and predict.\
Implemented Utils functions to calculate impurity, information gain, and undersampling the target class. \
Worked on handling the continuous features in the dataset, implementing and testing binning approach.\
Troubleshoot errors regarding building the tree for random forest\
Helped write parts of the report specifically checking the parts worked on above\
Re-tooled main class to allow for easy path specification for graders\
Wrote the README.md file
### Jeremy Middleman
Collaborated of the RandomForest class, namely implemented the bag function, re-tooled the other functions in the class\
Helped write parts of the report\
Debugged build_tree function in DTree.py to ensure that it fits with the RandomForest class
### Erick Yin
Worked jointly on early implementation of RandomForest class, namely build_forest function and aggregate_predictions (this was later appropriately re-tooled by Jeremy). Implemented calc_fraud_rate function in Utils.py as potential feature engineering, though this was later discarded in favor of other optimization methods. 
Wrote the majority of the report by following the rubric 
## Kaggle Run Summary:
Upon submission, our developed model achieved a balanced accuracy/Kaggle score of 67.792%. This metric represents the average recall from each class.
While a score of 0.67792 suggests reasonable success in instance identification, there is still room for improvement to potentially achieve a higher score. 
An example that we brainstormed was feature engineering. As discovered by our correlation matrix, a number of features were found to be seemingly irrelevant to the end findings. There was a potential to find a way to draw some more conclusions from those features had we decided to explore that avenue.
Date  run: 3/10/2024