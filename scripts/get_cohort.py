import pandas as pd 

# read in cohort
df = pd.read_csv("/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/raw/cohort.csv")

# drop duplicate ids
df_unique_ids = df.drop_duplicates(subset=['Subject'])

print(f"len unique ids is {len(df_unique_ids)}")
#take a random sample of 20 rows from each Group i.e. AD, MCI, EMCI, LMCI, CN
sample_cohort = df_unique_ids.groupby('Group', group_keys=False).apply(lambda x: x.sample(20))

#rest index
sample_cohort.reset_index(drop=True)

print(f"len sample_cohort df is {len(sample_cohort)}")
print(sample_cohort.nunique())

#save sample cohort
sample_cohort.to_csv("/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/raw/sample_cohort.csv", index=False)

# test = pd.read_csv("/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/raw/sample_cohort.csv")

# print(*test["Subject"], sep =', ')