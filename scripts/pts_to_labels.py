import pandas as pd 
from sklearn.model_selection import train_test_split

# read in cohort file
df = pd.read_csv("/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/raw/nii_sample_cohort.csv")
print(df.groupby("Group").size())

# Stratified split based on the "Group" column
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["Group"], random_state=42
)

# Add the "label" column
train_df["Label"] = "train"
test_df["Label"] = "test"

# Combine back into a single DataFrame
df = pd.concat([train_df, test_df]).reset_index(drop=True)

# Save the modified DataFrame to a new CSV file
output_file_path = "/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/raw/nii_sample_cohort_with_labels.csv"
df.to_csv(output_file_path, index=False)

print(f"Modified CSV file with 'label' column saved to: {output_file_path}")
print(df.groupby(["Group","Label"]).size())


