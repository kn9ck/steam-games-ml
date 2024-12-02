import pandas as pd

# Load the dataset
df = pd.read_csv('./datasets/descMediaReqSupp/steam.csv')

# Parse the owners column to get the lower bound
df['owners_lower'] = df['owners'].apply(lambda x: int(x.split('-')[0]))

# Convert categories to one-hot encoding
categories = df['categories'].str.get_dummies(sep=';')

# Add the one-hot encoded categories to the dataframe
df = pd.concat([df, categories], axis=1)

# Create a binary column indicating whether a game is successful
df['successful'] = df['owners_lower'] > 100000

import matplotlib.pyplot as plt
import seaborn as sns

# Analyze the distribution of categories
category_counts = categories.sum().sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=category_counts.values, y=category_counts.index)
plt.title('Distribution of Game Categories')
plt.xlabel('Count')
plt.ylabel('Category')
plt.show()

# Compare the distribution of categories between successful and unsuccessful games
successful_games = df[df['successful']]
unsuccessful_games = df[~df['successful']]

successful_category_counts = successful_games[categories.columns].sum().sort_values(ascending=False)
unsuccessful_category_counts = unsuccessful_games[categories.columns].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=successful_category_counts.values, y=successful_category_counts.index, color='blue', alpha=0.6, label='Successful')
sns.barplot(x=unsuccessful_category_counts.values, y=unsuccessful_category_counts.index, color='red', alpha=0.6, label='Unsuccessful')
plt.title('Comparison of Game Categories between Successful and Unsuccessful Games')
plt.xlabel('Count')
plt.ylabel('Category')
plt.legend()
plt.show()

from scipy.stats import chi2_contingency

# Perform chi-squared test for each category
chi2_results = {}
for category in categories.columns:
    contingency_table = pd.crosstab(df[category], df['successful'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    chi2_results[category] = p

# Filter categories with significant p-values
significant_categories = {k: v for k, v in chi2_results.items() if v < 0.05}
print("Significant Categories:", significant_categories)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Prepare the data for modeling
X = df[categories.columns]
y = df['successful']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=categories.columns).sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title('Feature Importance of Game Categories')
plt.xlabel('Importance')
plt.ylabel('Category')
plt.show()
