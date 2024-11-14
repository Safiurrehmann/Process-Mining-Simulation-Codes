import pandas as pd

# Load your data (replace with the path to your file)
df = pd.read_csv('overtimeNo2.csv')

# Filter rows where Job Satisfaction is 'Medium'
medium_job_satisfaction = df[df['Job Satisfaction'] == 'Medium']

# Save the filtered rows to a new CSV file
medium_job_satisfaction.to_csv('jobmedium.csv', index=False)

print("CSV file with Medium Job Satisfaction rows has been saved.")
