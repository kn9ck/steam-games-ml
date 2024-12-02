#create a visualization showing the success of games based on the release date such as the month rekleased
# Convert the release_date to datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import label encoder
from sklearn.preprocessing import LabelEncoder
#load the dataset
games = pd.read_csv('./datasets/descMediaReqSupp/steam.csv')

# Function to preprocess release date
def process_release_date(date_str):
    try:
        return pd.to_datetime(date_str)
    except:
        return np.nan

# Preprocess the data
def preprocess_data(df):
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()

    # Convert release_date to datetime
    df['release_date'] = df['release_date'].apply(process_release_date)

    # Convert english to int
    df['english'] = df['english'].astype(int)

    # Handle categorical variables using LabelEncoder
    le = LabelEncoder()
    df['developer'] = le.fit_transform(df['developer'].astype(str))
    df['publisher'] = le.fit_transform(df['publisher'].astype(str))

    # Convert platforms, categories, genres, and steamspy_tags to feature counts
    df['platforms'] = df['platforms'].str.count(';') + 1
    df['categories'] = df['categories'].str.count(';') + 1
    df['genres'] = df['genres'].str.count(';') + 1
    df['steamspy_tags'] = df['steamspy_tags'].str.count(';') + 1

    # Convert owners to binary success categories
    df['owners'] = df['owners'].apply(lambda x: 1 if int(x.split('-')[0]) > 500000 else 0)

    return df

# Prepare the features
def prepare_features(df):
    features = ['release_date', 'developer', 'publisher',
                'platforms', 'required_age', 'categories', 'genres', 'steamspy_tags',
                'achievements', 'price']

    X = df[features]
    y = df['owners']  # Target variable

    # Handle missing values
    X = X.fillna(X.mean())

    return X, y

# Preprocess the data
games = preprocess_data(games)

# Prepare the features
X, y = prepare_features(games)

# Visualize only successful games based on the release date
plt.figure(figsize=(12, 6))
successful_games = games[(games['release_date'].notna()) & (games['owners'] == 1)]
sns.countplot(data=successful_games, x=successful_games['release_date'].dt.month)
plt.title('Successful Games Based on Release Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()

#OWNERSHIP DISTRIBUTION PLOT
plt.figure(figsize=(12, 6))
sns.countplot(data=games, x='owners')
plt.title('Ownership Distribution')
plt.xlabel('Owners')
plt.ylabel('Count')
plt.xticks(range(2), ['Low Ownership', 'High Ownership'])
plt.show()

# After careful consideration, we retained the following features for our analysis:
#
#     Release date
#
#     Price
#
#     Developer/Publisher information
#
#     Genre categories
#
#     Platform support
#
#     Language support
#
#     Required age
#
#     Categories (e.g., single-player, multiplayer)
#
# [INSERT FEATURE CORRELATION HEATMAP]
#
# The correlation analysis of our selected features revealed interesting patterns in their relationships. Notably, we observed [describe key correlations found in the heatmap].

# correlation feature heatmap
numeric_columns = games.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(12, 6))
sns.heatmap(games[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()
