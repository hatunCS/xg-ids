# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Define file paths
DATA_DIR = "../dataset"  
TRAIN_FILE = "train.txt" 

# Column names for the NSL-KDD dataset
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficulty'
]


# Load in the dataset
train_path = os.path.join(DATA_DIR, TRAIN_FILE)
nslkdd = pd.read_csv(train_path, header=None, names=column_names)


# Displays preview of the dataset
print(f"Total instances: {nslkdd.shape[0]}")
print(f"Total Columns (41 features, 1 class label, and 1 difficulty ranking): {nslkdd.shape[1]}")
print("Preview of the first 5 instances of the NSL-KDD Dataset before processing:")
nslkdd.head()


# Dropping difficulty column (last column)
nslkdd = nslkdd.drop('difficulty', axis=1)


# Identify and track columns containing string data types (excluding the class label)
categorical_cols = []

for col in nslkdd.columns:
    if nslkdd[col].dtype == 'object' and col != 'class':
        categorical_cols.append(col)

print(f"Categorical columns found: {categorical_cols}")
print(f"\nUnique values per column:")
for col in categorical_cols:
    print(f"  {col}: {nslkdd[col].nunique()}")
print(f"\n__________________")



# Use pandas to convert categoricals + save the mappings
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    nslkdd[col] = le.fit_transform(nslkdd[col])
    label_encoders[col] = le

# Encode the class label separately
le_class = LabelEncoder()
nslkdd['class'] = le_class.fit_transform(nslkdd['class'])

print("Categorical columns encoded.")
print(f"Class mappings: {dict(zip(le_class.classes_, le_class.transform(le_class.classes_)))}")

# Initialize SMOTE
X = nslkdd.drop('class', axis=1)
y = nslkdd['class']

# Map class names to their encoded integers for sampling_strategy reference
class_map = dict(zip(le_class.classes_, le_class.transform(le_class.classes_)))
print("Class encoding map:")
print(class_map)

#SMOTE 1: Targeting Spy subcatagory (neighbor=1) + Generate: 5 +  Save results to ../dataset/nslkdd_SMOTE.txt
spy_encoded = class_map['spy']
current_count = (y == spy_encoded).sum()
smote1 = SMOTE(random_state=42, k_neighbors=1, sampling_strategy={spy_encoded: current_count + 5})
X_res, y_res = smote1.fit_resample(X, y)
df_checkpoint = pd.DataFrame(X_res, columns=X.columns)
df_checkpoint['class'] = y_res
df_checkpoint.to_csv('../dataset/nslkdd_SMOTE.txt', index=False)
print(f"SMOTE 1 done. spy: {current_count} → {current_count + 5}")


#SMOTE 2: Targeting perl subcatagory (neighbor=2) + Generate: 5 + Save results to ../dataset/nslkdd_SMOTE.txt
perl_encoded = class_map['perl']
current_count = (y_res == perl_encoded).sum()
smote2 = SMOTE(random_state=42, k_neighbors=2, sampling_strategy={perl_encoded: current_count + 5})
X_res, y_res = smote2.fit_resample(X_res, y_res)
df_checkpoint = pd.DataFrame(X_res, columns=X.columns)
df_checkpoint['class'] = y_res
df_checkpoint.to_csv('../dataset/nslkdd_SMOTE.txt', index=False)
print(f"SMOTE 2 done. perl: {current_count} → {current_count + 5}")

#SMOTE 3: Targeting phf subcatagory (neighbor=3) + Generate: 7 + Save results to ../dataset/nslkdd_SMOTE.txt
phf_encoded = class_map['phf']
current_count = (y_res == phf_encoded).sum()
smote3 = SMOTE(random_state=42, k_neighbors=3, sampling_strategy={phf_encoded: current_count + 7})
X_res, y_res = smote3.fit_resample(X_res, y_res)
df_checkpoint = pd.DataFrame(X_res, columns=X.columns)
df_checkpoint['class'] = y_res
df_checkpoint.to_csv('../dataset/nslkdd_SMOTE.txt', index=False)
print(f"SMOTE 3 done. phf: {current_count} → {current_count + 7}")


#SMOTE 4: Targeting multihop subcatagory (neighbor=5) + Generate: 15 + Save results to ../dataset/nslkdd_SMOTE.txt
multihop_encoded = class_map['multihop']
current_count = (y_res == multihop_encoded).sum()
smote4 = SMOTE(random_state=42, k_neighbors=5, sampling_strategy={multihop_encoded: current_count + 15})
X_res, y_res = smote4.fit_resample(X_res, y_res)
df_checkpoint = pd.DataFrame(X_res, columns=X.columns)
df_checkpoint['class'] = y_res
df_checkpoint.to_csv('../dataset/nslkdd_SMOTE.txt', index=False)
print(f"SMOTE 4 done. multihop: {current_count} → {current_count + 15}")


#SMOTE 5: Targeting ftp_write subcatagory (neighbor=5)+ Generate: 18 + Save results to ../dataset/nslkdd_SMOTE.txt
ftp_encoded = class_map['ftp_write']
current_count = (y_res == ftp_encoded).sum()
smote5 = SMOTE(random_state=42, k_neighbors=5, sampling_strategy={ftp_encoded: current_count + 18})
X_res, y_res = smote5.fit_resample(X_res, y_res)
df_checkpoint = pd.DataFrame(X_res, columns=X.columns)
df_checkpoint['class'] = y_res
df_checkpoint.to_csv('../dataset/nslkdd_SMOTE.txt', index=False)
print(f"SMOTE 5 done. ftp_write: {current_count} → {current_count + 18}")

#SMOTE 6: Targeting loadmodule subcatagory (neighbor=5) + Generate: 20 +  Save results to ../dataset/nslkdd_SMOTE.txt
loadmodule_encoded = class_map['loadmodule']
current_count = (y_res == loadmodule_encoded).sum()
smote6 = SMOTE(random_state=42, k_neighbors=5, sampling_strategy={loadmodule_encoded: current_count + 20})
X_res, y_res = smote6.fit_resample(X_res, y_res)
df_checkpoint = pd.DataFrame(X_res, columns=X.columns)
df_checkpoint['class'] = y_res
df_checkpoint.to_csv('../dataset/nslkdd_SMOTE.txt', index=False)
print(f"SMOTE 6 done. loadmodule: {current_count} → {current_count + 20}")

#SMOTE 7: Targeting rootkit subcatagory (neighbor=5) + Generate: 25 + Save results to ../dataset/nslkdd_SMOTE.txt
rootkit_encoded = class_map['rootkit']
current_count = (y_res == rootkit_encoded).sum()
smote7 = SMOTE(random_state=42, k_neighbors=5, sampling_strategy={rootkit_encoded: current_count + 25})
X_res, y_res = smote7.fit_resample(X_res, y_res)
df_checkpoint = pd.DataFrame(X_res, columns=X.columns)
df_checkpoint['class'] = y_res
df_checkpoint.to_csv('../dataset/nslkdd_SMOTE.txt', index=False)
print(f"SMOTE 7 done. rootkit: {current_count} → {current_count + 25}")


#SMOTE 8: Targeting imap subcatagory (neighbor=5) + Generate: 28 + Save results to ../dataset/nslkdd_SMOTE.txt
imap_encoded = class_map['imap']
current_count = (y_res == imap_encoded).sum()
smote8 = SMOTE(random_state=42, k_neighbors=5, sampling_strategy={imap_encoded: current_count + 28})
X_res, y_res = smote8.fit_resample(X_res, y_res)
df_checkpoint = pd.DataFrame(X_res, columns=X.columns)
df_checkpoint['class'] = y_res
df_checkpoint.to_csv('../dataset/nslkdd_SMOTE.txt', index=False)
print(f"SMOTE 8 done. imap: {current_count} → {current_count + 28}")

#SMOTE 9: Targeting land subcatagory (neighbor=5) + Generate: 48 + Save results to ../dataset/nslkdd_SMOTE.txt
land_encoded = class_map['land']
current_count = (y_res == land_encoded).sum()
smote9 = SMOTE(random_state=42, k_neighbors=5, sampling_strategy={land_encoded: current_count + 48})
X_res, y_res = smote9.fit_resample(X_res, y_res)
df_checkpoint = pd.DataFrame(X_res, columns=X.columns)
df_checkpoint['class'] = y_res
df_checkpoint.to_csv('../dataset/nslkdd_SMOTE.txt', index=False)
print(f"SMOTE 9 done. land: {current_count} → {current_count + 48}")

#SMOTE 10: Targeting warezmaster subcatagory(neighbor=5) + Generate: 50 + Save results to ../dataset/nslkdd_SMOTE.txt
warezmaster_encoded = class_map['warezmaster']
current_count = (y_res == warezmaster_encoded).sum()
smote10 = SMOTE(random_state=42, k_neighbors=5, sampling_strategy={warezmaster_encoded: current_count + 50})
X_res, y_res = smote10.fit_resample(X_res, y_res)
df_checkpoint = pd.DataFrame(X_res, columns=X.columns)
df_checkpoint['class'] = y_res
df_checkpoint.to_csv('../dataset/nslkdd_SMOTE.txt', index=False)
print(f"SMOTE 10 done. warezmaster: {current_count} → {current_count + 50}")

#SMOTE 11: Targeting buffer_overflow subcatagory(neighbor=5) + Generate: 70 + Save results to ../dataset/nslkdd_SMOTE.txt
buffer_overflow_encoded = class_map['buffer_overflow']
current_count = (y_res == buffer_overflow_encoded).sum()
smote11 = SMOTE(random_state=42, k_neighbors=5, sampling_strategy={buffer_overflow_encoded: current_count + 70})
X_res, y_res = smote11.fit_resample(X_res, y_res)
df_checkpoint = pd.DataFrame(X_res, columns=X.columns)
df_checkpoint['class'] = y_res
df_checkpoint.to_csv('../dataset/nslkdd_SMOTE.txt', index=False)
print(f"SMOTE 11 done. buffer_overflow: {current_count} → {current_count + 70}")

#SMOTE 12: Targeting guess_passwd subcatagory (neighbor=5) + Generate: 100 + Save results to ../dataset/nslkdd_SMOTE.txt
guess_passwd_encoded = class_map['guess_passwd']
current_count = (y_res == guess_passwd_encoded).sum()
smote12 = SMOTE(random_state=42, k_neighbors=5, sampling_strategy={guess_passwd_encoded: current_count + 100})
X_res, y_res = smote12.fit_resample(X_res, y_res)
df_checkpoint = pd.DataFrame(X_res, columns=X.columns)
df_checkpoint['class'] = y_res
df_checkpoint.to_csv('../dataset/nslkdd_SMOTE.txt', index=False)
print(f"SMOTE 12 done. guess_passwd: {current_count} → {current_count + 100}")

#SMOTE 13: Targeting pod subcatagory (neighbor=5) + Generate: 200 + Save results to ../dataset/nslkdd_SMOTE.txt
pod_encoded = class_map['pod']
current_count = (y_res == pod_encoded).sum()
smote13 = SMOTE(random_state=42, k_neighbors=5, sampling_strategy={pod_encoded: current_count + 200})
X_res, y_res = smote13.fit_resample(X_res, y_res)
df_checkpoint = pd.DataFrame(X_res, columns=X.columns)
df_checkpoint['class'] = y_res
df_checkpoint.to_csv('../dataset/nslkdd_SMOTE.txt', index=False)
print(f"SMOTE 13 done. pod: {current_count} → {current_count + 200}")


#SMOTE 14: Targeting warezclient subcatagory (neighbor=5) + Generate: 200 + Save results to ../dataset/nslkdd_SMOTE.txt
warezclient_encoded = class_map['warezclient']
current_count = (y_res == warezclient_encoded).sum()
smote14 = SMOTE(random_state=42, k_neighbors=5, sampling_strategy={warezclient_encoded: current_count + 200})
X_res, y_res = smote14.fit_resample(X_res, y_res)
df_checkpoint = pd.DataFrame(X_res, columns=X.columns)
df_checkpoint['class'] = y_res
df_checkpoint.to_csv('../dataset/nslkdd_SMOTE.txt', index=False)
print(f"SMOTE 14 done. warezclient: {current_count} → {current_count + 200}")

#SMOTE 15: Targeting teardrop subcatagory (neighbor=5) + Generate: 200 + Save results to ../dataset/nslkdd_SMOTE.txt
teardrop_encoded = class_map['teardrop']
current_count = (y_res == teardrop_encoded).sum()
smote15 = SMOTE(random_state=42, k_neighbors=5, sampling_strategy={teardrop_encoded: current_count + 200})
X_res, y_res = smote15.fit_resample(X_res, y_res)
df_checkpoint = pd.DataFrame(X_res, columns=X.columns)
df_checkpoint['class'] = y_res
df_checkpoint.to_csv('../dataset/nslkdd_SMOTE.txt', index=False)
print(f"SMOTE 15 done. teardrop: {current_count} → {current_count + 200}")

#SMOTE 16: Targeting back subcatagory (neighbor=5) + Generate: 200 + Save results to ../dataset/nslkdd_SMOTE.txt
back_encoded = class_map['back']
current_count = (y_res == back_encoded).sum()
smote16 = SMOTE(random_state=42, k_neighbors=5, sampling_strategy={back_encoded: current_count + 200})
X_res, y_res = smote16.fit_resample(X_res, y_res)
df_checkpoint = pd.DataFrame(X_res, columns=X.columns)
df_checkpoint['class'] = y_res
df_checkpoint.to_csv('../dataset/nslkdd_SMOTE.txt', index=False)
print(f"SMOTE 16 done. back: {current_count} → {current_count + 200}")



# Sanity Check
# Hardcode quantity of each subcategory PRE-SMOTE
original_subcategory_quantity = {
    'back': 956, 'buffer_overflow': 30, 'ftp_write': 8, 'guess_passwd': 53,
    'imap': 11, 'ipsweep': 3599, 'land': 18, 'loadmodule': 9, 'multihop': 7,
    'neptune': 41214, 'nmap': 1493, 'normal': 67343, 'perl': 3, 'phf': 4,
    'pod': 201, 'portsweep': 2931, 'rootkit': 10, 'satan': 3633, 'smurf': 2646,
    'spy': 2, 'teardrop': 892, 'warezclient': 890, 'warezmaster': 20
}

# Count quantity of each subcategory POST-SMOTE
decoded_classes = le_class.inverse_transform(y_res)
post_smote_subcategory_quantity = dict(pd.Series(decoded_classes).value_counts())

# Check the nslkdd_SMOTE.txt file for NaN/null values
nan_count = pd.DataFrame(X_res).isnull().sum().sum()

# Feature column count
feature_column_count = X_res.shape[1]

# Display results table
print(f"Feature column count: {feature_column_count}")
print(f"NaN/Null values found: {nan_count}\n")

print(f"{'Subcategory':<20} {'Original':>10} {'Post-SMOTE':>12} {'Delta':>8}")
print("-" * 52)
for cls in sorted(original_subcategory_quantity.keys()):
    original = original_subcategory_quantity[cls]
    post = post_smote_subcategory_quantity.get(cls, 0)
    delta = post - original
    print(f"{cls:<20} {original:>10} {post:>12} {delta:>+8}")

total_original = sum(original_subcategory_quantity.values())
total_post = len(y_res)
print("-" * 52)
print(f"{'TOTAL':<20} {total_original:>10} {total_post:>12} {total_post - total_original:>+8}")


# TODO: Decode the class labels in the nslkdd_SMOTE.txt file back to their original string values using the saved LabelEncoder mappings.
df_final = pd.DataFrame(X_res, columns=X.columns)
df_final['class'] = y_res

for col in categorical_cols:
    df_final[col] = label_encoders[col].inverse_transform(df_final[col].astype(int))
df_final['class'] = le_class.inverse_transform(df_final['class'])

# TODO: Export the newly balanced dataset to a new file called "verified_nslkdd_SMOTE.txt”
df_final.to_csv('../dataset/verified_nslkdd_SMOTE.txt', index=False)
print(f"Export complete. Total instances: {df_final.shape[0]}")