#This file contains code designed to predict passenger survival on the Titanic as part of a Kaggle competition. 
#The model achieves an accuracy of 87%.

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# Loading the training data
train_file_path = "C:/Users/Ildar Shakirzianov/Downloads/titanic/train.csv"
data_train = pd.read_csv(train_file_path)

# Filling in the blanks for the age column with median values
data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())

# Cabin feature engineering
def cabin_letter_extractor(cabin):
    if pd.notna(cabin):
        match = re.search(r'[^\d]+', cabin)
        return match.group(0).strip() if match else ''
    else:
        return ''

def cabin_number_extractor(cabin):
    if pd.notna(cabin):
        match = re.search(r'\d+', cabin)
        return match.group(0).strip() if match else ''
    else:
        return ''

data_train['Cabin Letter'] = data_train['Cabin'].apply(cabin_letter_extractor)
data_train['Cabin Number'] = data_train['Cabin'].apply(cabin_number_extractor)

# Ticket feature engineering
def ticket_prefix_extractor(ticket):
    if pd.notna(ticket):
        match = re.match(r'^[^\d\s/]+', ticket)
        return match.group(0).strip() if match else ''  # Return prefix or empty string
    else:
        return ''

def ticket_number_extractor(ticket):
    if not ticket[0].isdigit():
        return ticket.split(' ', 1)[1] if ' ' in ticket else ''
    return ticket

data_train['Ticket Prefix'] = data_train['Ticket'].apply(ticket_prefix_extractor)
data_train['Ticket Number'] = data_train['Ticket'].apply(ticket_number_extractor)

# Ensuring numerical columns for ticket number
data_train['Ticket Number'] = pd.to_numeric(data_train['Ticket Number'], errors='coerce')


cabin_counts = data_train['Cabin Letter'].value_counts()
threshold = 5
rare_cabins = cabin_counts[cabin_counts < threshold].index
data_train['Cabin Letter'] = data_train['Cabin Letter'].apply(lambda x: x if x not in rare_cabins else 'Other')

# Handling categorical variables and rare values
data_train = pd.get_dummies(data_train, columns=['Sex', 'Embarked'], drop_first=True)


# Filtering out rows with missing cabin data
data_train_without_cabin_blanks = data_train[data_train['Cabin'].notna()]

# Label encoding for cabin Letters
y = data_train_without_cabin_blanks['Cabin Letter']
le = LabelEncoder()
y = le.fit_transform(y)

# Selecting features for model
X = data_train_without_cabin_blanks[['Pclass', 'Fare', 'SibSp', 'Parch']]

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Filling missing values with median
X_train = X_train.fillna(X_train.median())

# Rebalancing the sample
ros = RandomOverSampler(sampling_strategy='auto', random_state=41)
X_train, y_train = ros.fit_resample(X_train, y_train)

# Defining parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of boosting rounds
    'learning_rate': [0.01, 0.05, 0.1],  # Learning rate
    'max_depth': [3, 5, 7],  # Depth of trees
    'subsample': [0.8, 1.0],  # Subsample ratio of training data
    'colsample_bytree': [0.8, 1.0]  # Fraction of features to use for each tree
}

# Initialize XGBoost and GridSearchCV
xgb_cabin = XGBClassifier(random_state=41, eval_metric='mlogloss')
grid_search_cabin = GridSearchCV(estimator=xgb_cabin, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# Fitting the model using GridSearchCV
grid_search_cabin.fit(X_train, y_train)

# Getting the best estimator from GridSearchCV
best_xgb_cabin = grid_search_cabin.best_estimator_

# Predict missing cabin letters
data_with_missing_cabins = data_train[data_train['Cabin'].isna()]
X_missing = data_with_missing_cabins[['Pclass', 'Fare', 'SibSp', 'Parch']]
X_missing = X_missing.fillna(X_train.median())  # Fill missing values with median
y_missing_pred = best_xgb_cabin.predict(X_missing)

# Converting predicted labels back to cabin letters
cabin_letters_pred = le.inverse_transform(y_missing_pred)
data_with_missing_cabins['Cabin Letter'] = cabin_letters_pred

# Updating the original dataset with the filled-in cabin letters
data_train.loc[data_with_missing_cabins.index, 'Cabin Letter'] = data_with_missing_cabins['Cabin Letter']

# Preparing data for training the survival model
y = data_train['Survived']
X = data_train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin Letter', 'Sex_male',
                'Embarked_Q', 'Embarked_S']]

# Encode cabin letters and handle missing values
X = pd.get_dummies(X, columns=['Cabin Letter'], drop_first=True)
X['Fare'] = X['Fare'].fillna(X['Fare'].median())
X['Family size'] = X['SibSp']+X['Parch']+1

X = X.drop(columns = ['Cabin Letter_C', 'Cabin Letter_D', 'Cabin Letter_D', 'Cabin Letter_B', 'Cabin Letter_F',
                                          'Cabin Letter_Other', 'SibSp', 'Parch', 'Embarked_S', 'Embarked_Q'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Scaling numerical features 
scaler = StandardScaler()
X_train[['Age']] = scaler.fit_transform(X_train[['Age']])
X_test[['Age']] = scaler.transform(X_test[['Age']])

# Rebalancing the sample
ros = RandomOverSampler(sampling_strategy='auto', random_state=41)
X_train, y_train = ros.fit_resample(X_train, y_train)

# Fitting the base models
xgb = XGBClassifier(random_state=41, eval_metric='mlogloss')
catboost = CatBoostClassifier(random_seed=41, verbose=0)

# Training both models
xgb.fit(X_train, y_train)
catboost.fit(X_train, y_train)
xgb_probs = xgb.predict_proba(X_test)[:, 1]
catboost_probs = catboost.predict_proba(X_test)[:, 1]

# Averaging the probabilities
combined_probs = (xgb_probs + catboost_probs) / 2
combined_pred = (combined_probs >= 0.5).astype(int)

# Evaluating Averaged Model
accuracy = accuracy_score(y_test, combined_pred)
roc_auc = roc_auc_score(y_test, combined_probs)
print(f"Averaged Model Accuracy: {accuracy:.2f}")
print(f"Averaged Model ROC AUC: {roc_auc:.2f}")
print("Averaged Model Classification Report:")
print(classification_report(y_test, combined_pred))

# Stacking
base_models = [
    ('xgb', XGBClassifier(random_state=41, eval_metric='mlogloss')),
    ('catboost', CatBoostClassifier(random_seed=41, verbose=0))
]

stacked_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=5)
stacked_model.fit(X_train, y_train)

# Evaluating the stacked model
stacked_pred = stacked_model.predict(X_test)
stacked_accuracy = accuracy_score(y_test, stacked_pred)
stacked_roc_auc = roc_auc_score(y_test, stacked_model.predict_proba(X_test)[:, 1])

print(f"Stacked Model Accuracy: {stacked_accuracy:.2f}")
print(f"Stacked Model ROC AUC: {stacked_roc_auc:.2f}")
print("Stacked Model Classification Report:")
print(classification_report(y_test, stacked_pred))

# Feature importance
importances = catboost.get_feature_importance()
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("Feature Importance from CatBoost:")
print(importance_df)

# Loading the test dataset
test_file_path = "C:/Users/Ildar Shakirzianov/Downloads/titanic/test.csv"
data_test = pd.read_csv(test_file_path)
data_test['Age'] = data_test['Age'].fillna(data_test['Age'].median())
data_test['Cabin Letter'] = data_test['Cabin'].apply(cabin_letter_extractor)
data_test['Cabin Number'] = data_test['Cabin'].apply(cabin_number_extractor)
data_test['Ticket Prefix'] = data_test['Ticket'].apply(ticket_prefix_extractor)
data_test['Ticket Number'] = data_test['Ticket'].apply(ticket_number_extractor)
data_test['Ticket Number'] = pd.to_numeric(data_test['Ticket Number'], errors='coerce')

cabin_counts = data_test['Cabin Letter'].value_counts()
rare_cabins = cabin_counts[cabin_counts < threshold].index
data_test['Cabin Letter'] = data_test['Cabin Letter'].apply(lambda x: x if x not in rare_cabins else 'Other')


data_test = pd.get_dummies(data_test, columns=['Sex', 'Embarked'], drop_first=True)

data_test_without_cabin_blanks = data_test[data_test['Cabin'].notna()]

# Encoding cabin letters
y = data_test_without_cabin_blanks['Cabin Letter']
le_1 = LabelEncoder()
y = le_1.fit_transform(y)

# Predicting the missing cabin letters
data_with_missing_cabins = data_test[data_test['Cabin'].isna()]
X_missing = data_with_missing_cabins[['Pclass', 'Fare', 'SibSp', 'Parch']]
X_missing = X_missing.fillna(X_train.median())  
y_missing_pred = best_xgb_cabin.predict(X_missing)

# Checking if predicted labels are within the known labels from the encoder
try:
    cabin_letters_pred = le_1.inverse_transform(y_missing_pred)
    data_with_missing_cabins['Cabin Letter'] = cabin_letters_pred
except ValueError as e:
    print("Error in inverse_transform:", e)
    unseen_labels = set(y_missing_pred) - set(le_1.classes_)
    print("Unseen labels in prediction:", unseen_labels)
    data_with_missing_cabins['Cabin Letter'] = 'Other'

# Updating the original dataset
data_test.loc[data_with_missing_cabins.index, 'Cabin Letter'] = data_with_missing_cabins['Cabin Letter']

X_test_data = data_test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin Letter', 'Sex_male',
                         'Embarked_Q', 'Embarked_S']]
X_test_data = pd.get_dummies(X_test_data, columns=['Cabin Letter'], drop_first=True)
X_test_data['Fare'] = X_test_data['Fare'].fillna(X_test_data['Fare'].median())
X_test_data[['Age']] = scaler.transform(X_test_data[['Age']])
X_test_data['Family size'] = X_test_data['SibSp']+X_test_data['Parch']+1

X_test_data = X_test_data.drop(columns = ['Cabin Letter_C', 'Cabin Letter_D', 'Cabin Letter_D', 'Cabin Letter_B', 'Cabin Letter_F',
                                          'Cabin Letter_Other', 'SibSp', 'Parch', 'Embarked_S', 'Embarked_Q'])
# Predicting survival for the test set using the stacked model
y_test_pred = stacked_model.predict(X_test_data) 
data_test['Survived'] = y_test_pred

output_file_path = "C:/Users/Ildar Shakirzianov/Downloads/titanic/test_with_predictions.csv"
data_test[['PassengerId', 'Survived']].to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")
