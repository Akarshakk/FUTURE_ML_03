import pandas as pd
import random

input_file = 'job_descriptions.csv'
output_file = 'sample_job_descriptions.csv'
sample_size = 2000

# Try to read only the necessary amount or sample
# Since the file is 1.7GB, we can read chunks, or use pandas sample. 
# A simple way without loading everything is reading the first 5000 rows and sampling from there, 
# or just taking the first 2000 rows to ensure it runs quickly. Let's just take the first 5000 rows to be safe.
print(f"Reading first 5000 rows from {input_file}...")
try:
    df = pd.read_csv(input_file, nrows=5000)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Take sample_size
    sample_df = df.head(sample_size)
    
    sample_df.to_csv(output_file, index=False)
    print(f"Successfully created {output_file} with {len(sample_df)} records.")
except Exception as e:
    print(f"Error: {e}")
