import pandas as pd
from Utils import undersample, entropy, bin_all_columns
from sklearn.model_selection import train_test_split
import DTree
from RandomForest import RandomForest
# import time


training_path = "resources/clened_train.csv"  # add path to training data
testing_path = "resources/cleaned_test.csv"  # add path to testing data
# Read the data from the csv file using pandas
test_data = pd.read_csv('resources/test.csv')

# sample datasets
forest = RandomForest(entropy, "isFraud")
data = pd.read_csv(training_path)
data['card3'] = pd.to_numeric(data['card3'], errors='coerce')

# split training data into 3/4th training, 1/4th validation
X = data.drop(columns=['isFraud'])
y = data["isFraud"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = .25, stratify = y)

validation_data = X_val

X_train, y_train = undersample(X_train, y_train)

# build forest on training data and use the forest to make predictions for validation data
forest.build_forest(X_train, y_train)
forest.print_forest()
prediction_list = forest.aggregate_predictions(validation_data)
y_val_array = y_val.to_numpy()

correct_predictions = sum(prediction_list == y_val_array)

print(f'Number of correct predictions: {correct_predictions} out of {len(y_val)}')

accuracy = correct_predictions / len(y_val)
print(f'Accuracy: {accuracy:.4f}')

submit_data = pd.read_csv(testing_path)
transaction_ID = pd.DataFrame(test_data, columns=['TransactionID'])
submit_data['card3'] = pd.to_numeric(submit_data['card3'], errors='coerce')

prediction_list = forest.aggregate_predictions(submit_data)
prediction_list = pd.DataFrame(prediction_list, columns=['isFraud'])
prediction_list = transaction_ID.join(prediction_list)
prediction_list.to_csv('resources/submission.csv', index=False)
