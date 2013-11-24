# Load modules
import pandas as pd
from sklearn import tree, metrics

# Load in data and create training and test sets. dropping all na columns, for kicks.
l_train = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/lemons/lemon_training.csv')
l_test = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/lemons/lemon_test.csv')
l_train = l_train.dropna(axis=1)
l_test = l_test.dropna(axis=1)

# Generating a list of continuous data features from the describe dataframe. 
# Then, removing the two non-features (RefId is an index, IsBadBuy is the prediction value)
features = list(l_train.describe().columns)
features.remove('RefId')
features.remove('IsBadBuy')

# Creating the actual training and test sets. Yours should cross validate on the training data.
train_X = l_train[features].values
train_y = l_train.IsBadBuy.values
test_X = l_test[features].values
test_y = l_test.IsBadBuy.values

# Create a classifier and prediction.
clf = tree.DecisionTreeClassifier().fit(train_X, train_y)
clf.score(train_X, train_y)
y_pred = clf.predict(test_X)


metrics.roc_auc_score(train_y, clf.predict(train_X))

# Create a submission
submission = pd.DataFrame({ 'RefId' : l_test.RefId, 'prediction' : y_pred })
submission.to_csv('submission.csv')