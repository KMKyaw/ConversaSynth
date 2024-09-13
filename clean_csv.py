import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('annotations.csv')

# Define a function to remove "conversation" from the filename
def clean_filename(filename):
    # Split the filename to extract the number part
    parts = filename.split('conversation')
    if len(parts) > 1:
        return parts[1]
    return filename

# Apply the function to the 'filename' column
df['filename'] = df['filename'].apply(clean_filename)

# Save the updated DataFrame to a new CSV file
df.to_csv('annotations.csv', index=False)
