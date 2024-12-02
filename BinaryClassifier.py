import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

#Load the data
df = pd.read_csv("./datasets/descMediaReqSupp/steam.csv")

#Create the target variable (1 if owners > 500000, 0 otherwise)
df['success'] = df['owners'].apply(lambda x: 1 if int(x.split('-')[0]) > 500000 else 0)

# Select features for the model
# We'll use numerical features and encode categorical ones
features_to_use = ['required_age', 'achievements', 'positive_ratings',
                   'negative_ratings', 'average_playtime', 'median_playtime',
                   'price', 'genres', 'categories', 'steamspy_tags']

#Create feature dataframe
X = df[features_to_use].copy()

#Handle categorical features
le = LabelEncoder()
categorical_features = ['genres', 'categories', 'steamspy_tags']
for feature in categorical_features:
    X[feature] = le.fit_transform(X[feature].astype(str))

#Create target variable
y = df['success']

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train the model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

#Make predictions
y_pred = rf_classifier.predict(X_test_scaled)

#Print results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Feature importance
feature_importance = pd.DataFrame({
    'feature': features_to_use,
    'importance': rf_classifier.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)


#Create feature importance plot
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xticks(rotation=45, ha='right')
plt.title('Feature Importance in Game Success Prediction')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

#Alternative visualization using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Game Success Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

#Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

#Create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.text(-0.4, -0.4, '0 = Unsuccessful\n1 = Successful')
plt.tight_layout()
plt.show()
