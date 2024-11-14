import pandas as pd
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Function to calculate entropy
def entropy(labels):
    label_counts = Counter(labels)
    total_count = len(labels)
    probs = [count / total_count for count in label_counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)

# Function to calculate information gain
def information_gain(X, y, feature):
    original_entropy = entropy(y)
    unique_vals, counts = np.unique(X[feature], return_counts=True)
    
    weighted_entropy = 0
    for i, val in enumerate(unique_vals):
        subset_y = y[X[feature] == val]
        weighted_entropy += (counts[i] / len(X)) * entropy(subset_y)
    
    return original_entropy - weighted_entropy

# Read the data from the CSV file
df = pd.read_csv('assignmentdata.csv')

# Mapping categorical variables to numerical values
gender_map = {'Male': 0, 'Female': 1}
df['Gender'] = df['Gender'].map(gender_map)

work_life_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
df['Work-Life Balance'] = df['Work-Life Balance'].map(work_life_map)

job_satisfaction_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}
df['Job Satisfaction'] = df['Job Satisfaction'].map(job_satisfaction_map)

performance_rating_map = {'Below Average': 0, 'Average': 1, 'High': 2, 'Very High': 3}
df['Performance Rating'] = df['Performance Rating'].map(performance_rating_map)

education_level_map = {
    'High School': 0, 
    'Associate Degree': 1, 
    'Bachelor’s Degree': 2, 
    'Master’s Degree': 3, 
    'PhD': 4
}
df['Education Level'] = df['Education Level'].map(education_level_map)

attrition_map = {'Stayed': 0, 'Left': 1}
df['Attrition'] = df['Attrition'].map(attrition_map)

# Feature selection
features = ['Age', 'Years at Company', 'Monthly Income', 'Work-Life Balance', 
            'Job Satisfaction', 'Performance Rating', 'Number of Promotions', 
            'Distance from Home', 'Education Level']

X = df[features]
y = df['Attrition']

# Calculate and print entropy for the response variable (Attrition)
response_entropy = entropy(y)
print(f"Entropy of response variable (Attrition): {response_entropy:.4f}")

# Calculate and print information gain for each predictor variable
for feature in features:
    gain = information_gain(df, y, feature)
    print(f"Information gain for {feature}: {gain:.4f}")

# Train the Decision Tree Classifier
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(dtree, feature_names=features, class_names=['Stayed', 'Left'], filled=True)
plt.show()
