import os
import pandas as pd

df = pd.read_csv("Database/trainingdata_basic.csv")

for index, rows in df.iterrows():
    old_path = rows['Images']
    label = rows['Label']

    directory = os.path.dirname(old_path)
    filename = os.path.basename(old_path)

    # split filename and extension
    name, ext = os.path.splitext(filename)
    # append to new filename
    new_filename = f"{name}_{label}{ext}" 
    new_path = os.path.join(directory, new_filename)

    try:
        os.rename(old_path, new_path)
        print(f"Rename {old_path} to {new_path}")
    except FileNotFoundError:
        print(f"File not found: {old_path}")
    except Exception as e:
        print(f"Error renaming {old_path}. Exception occurred: {str(e)}")