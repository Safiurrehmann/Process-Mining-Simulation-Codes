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

# Function to count the occurrences of each category in a feature
def count_categories(feature):
    category_counts = dict(Counter(feature))
    return category_counts

# Function to count 'Stayed' and 'Left' in each category of a feature
def count_attrition_in_categories(feature, attrition):
    category_attrition_counts = {}
    unique_vals = np.unique(feature)

    for val in unique_vals:
        stayed_count = sum((feature == val) & (attrition == 0))
        left_count = sum((feature == val) & (attrition == 1))
        category_attrition_counts[val] = {'Stayed': stayed_count, 'Left': left_count}

    return category_attrition_counts


def weightedentropy(feature):
    unique_vals, counts = np.unique(feature, return_counts=True)
    weighted_entropy = 0
    for i, val in enumerate(unique_vals):
        subset_y = y[feature == val]
        weighted_entropy += (counts[i] / len(feature)) * entropy(subset_y)

    return weighted_entropy

def category_entropy(feature, target):
    unique_vals = np.unique(feature)
    category_entropies = {}

    for val in unique_vals:
        subset_y = target[feature == val]
        category_entropies[val] = entropy(subset_y)

    return category_entropies

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
df = pd.read_csv('jobmedium.csv')

# Mapping categorical variables to numerical values
overtime = {'Yes': 0, 'No': 1}
df['Overtime'] = df['Overtime'].map(overtime)

opportunities = {'Yes': 0, 'No': 1}
df['Innovation Opportunities'] = df['Innovation Opportunities'].map(opportunities)

work_life_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
df['Work-Life Balance'] = df['Work-Life Balance'].map(work_life_map)

job_satisfaction_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}
df['Job Satisfaction'] = df['Job Satisfaction'].map(job_satisfaction_map)

performance_rating_map = {'Low': 0, 'Below Average': 1, 'Average': 2, 'High': 3}
df['Performance Rating'] = df['Performance Rating'].map(performance_rating_map)

attrition_map = {'Stayed': 0, 'Left': 1}
df['Attrition'] = df['Attrition'].map(attrition_map)

# Feature selection
features = ['Work-Life Balance', 'Job Satisfaction', 'Performance Rating', 'Innovation Opportunities', 
            'Overtime']

X = df[features]
y = df['Attrition']
w=df['Work-Life Balance']
j=df['Job Satisfaction']
p=df['Performance Rating']
over=df['Overtime']
opp=df['Innovation Opportunities']

# Calculate and print entropy for the response variable (Attrition)
response_entropy = entropy(y)
print(f"Entropy of response variable (Attrition): {response_entropy:.4f}")
wlb = weightedentropy(w)
print(f"Entropy of response variable (Work-Life Balance): {wlb:.4f}")
js = weightedentropy(j)
print(f"Entropy of response variable (Job Satisfaction): {js:.4f}")
pr = weightedentropy(p)
print(f"Entropy of response variable (Performance Rating): {pr:.4f}")
ot = weightedentropy(over)
print(f"Entropy of response variable (Overtime): {ot:.4f}")
io = weightedentropy(opp)
print(f"Entropy of response variable (Innovation Opportunities): {io:.4f}")


print("\n")

# Counting categories for 'Work-Life Balance'
work_life_counts = count_categories(df['Work-Life Balance'])
print("Category counts in Work-Life Balance:")
for category, count in work_life_counts.items():
    print(f"Category {category}: {count}")

# Counting categories for 'Job Satisfaction'
job_satisfaction_counts = count_categories(df['Job Satisfaction'])
print("\nCategory counts in Job Satisfaction:")
for category, count in job_satisfaction_counts.items():
    print(f"Category {category}: {count}")

# Counting categories for 'Performance Rating'
performance_rating_counts = count_categories(df['Performance Rating'])
print("\nCategory counts in Performance Rating:")
for category, count in performance_rating_counts.items():
    print(f"Category {category}: {count}")

# Counting categories for 'Overtime'
overtime_counts = count_categories(df['Overtime'])
print("\nCategory counts in Overtime:")
for category, count in overtime_counts.items():
    print(f"Category {category}: {count}")

# Counting categories for 'Innovation Opportunities'
innovation_opportunities_counts = count_categories(df['Innovation Opportunities'])
print("\nCategory counts in Innovation Opportunities:")
for category, count in innovation_opportunities_counts.items():
    print(f"Category {category}: {count}")



print("\n")
work_life_entropy = category_entropy(df['Work-Life Balance'], y)
print("Entropy for each category in Work-Life Balance:")
for category, ent in work_life_entropy.items():
    print(f"Category {category}: {ent:.4f}")


print("\n")
job_satisfaction_entropy = category_entropy(df['Job Satisfaction'], y)
print("Entropy for each category in Job Satisfaction:")
for category, ent in job_satisfaction_entropy.items():
    print(f"Category {category}: {ent:.4f}")


print("\n")
# Calling the category_entropy function for 'Performance Rating'
performance_rating_entropy = category_entropy(df['Performance Rating'], y)
print("Entropy for each category in Performance Rating:")
for category, ent in performance_rating_entropy.items():
    print(f"Category {category}: {ent:.4f}")

# Calling the category_entropy function for 'Overtime'
overtime_entropy = category_entropy(df['Overtime'], y)
print("\nEntropy for each category in Overtime:")
for category, ent in overtime_entropy.items():
    print(f"Category {category}: {ent:.4f}")

# Calling the category_entropy function for 'Innovation Opportunities'
innovation_opportunities_entropy = category_entropy(df['Innovation Opportunities'], y)
print("\nEntropy for each category in Innovation Opportunities:")
for category, ent in innovation_opportunities_entropy.items():
    print(f"Category {category}: {ent:.4f}")


print("\n")
# Calculate and print information gain for each predictor variable
for feature in features:
    gain = information_gain(df, y, feature)
    print(f"Information gain for {feature}: {gain:.4f}")

print("\n")

# Counting 'Stayed' and 'Left' for 'Work-Life Balance'
work_life_attrition_counts = count_attrition_in_categories(df['Work-Life Balance'], df['Attrition'])
print("Attrition counts in Work-Life Balance:")
for category, counts in work_life_attrition_counts.items():
    print(f"Category {category}: Stayed = {counts['Stayed']}, Left = {counts['Left']}")

# Counting 'Stayed' and 'Left' for 'Job Satisfaction'
job_satisfaction_attrition_counts = count_attrition_in_categories(df['Job Satisfaction'], df['Attrition'])
print("\nAttrition counts in Job Satisfaction:")
for category, counts in job_satisfaction_attrition_counts.items():
    print(f"Category {category}: Stayed = {counts['Stayed']}, Left = {counts['Left']}")

# Counting 'Stayed' and 'Left' for 'Performance Rating'
performance_rating_attrition_counts = count_attrition_in_categories(df['Performance Rating'], df['Attrition'])
print("\nAttrition counts in Performance Rating:")
for category, counts in performance_rating_attrition_counts.items():
    print(f"Category {category}: Stayed = {counts['Stayed']}, Left = {counts['Left']}")

# Counting 'Stayed' and 'Left' for 'Overtime'
overtime_attrition_counts = count_attrition_in_categories(df['Overtime'], df['Attrition'])
print("\nAttrition counts in Overtime:")
for category, counts in overtime_attrition_counts.items():
    print(f"Category {category}: Stayed = {counts['Stayed']}, Left = {counts['Left']}")

# Counting 'Stayed' and 'Left' for 'Innovation Opportunities'
innovation_opportunities_attrition_counts = count_attrition_in_categories(df['Innovation Opportunities'], df['Attrition'])
print("\nAttrition counts in Innovation Opportunities:")
for category, counts in innovation_opportunities_attrition_counts.items():
    print(f"Category {category}: Stayed = {counts['Stayed']}, Left = {counts['Left']}")


# Train the Decision Tree Classifier
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(dtree, feature_names=features, class_names=['Stayed', 'Left'], filled=True)
plt.show()
