import csv
import pandas as pd

filepath_test = "/Users/glenchua/Documents/thesis_project/Database/testdata_basic.csv"
filepath_train = "/Users/glenchua/Documents/thesis_project/Database/trainingdata_basic.csv"
rf_test = "Database/basic/Image/aligned/test_"
rf_train = "Database/basic/Image/aligned/train_"
# train_data = []
# test_data = []

# with open("/Users/glenchua/Documents/thesis_project/Database/compound/EmoLabel/list_patition_label.txt", "r") as f:
#     for line in f:
#         filename, label = line.strip().split()
#         label = int(label)

#         if 'test' in filename:
#             test_data.append((filename, label))
#         elif 'train' in filename:
#             train_data.append((filename, label))

# with open("compound_test_data.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Image", "Label"])
#     writer.writerows(test_data)

# with open("compound_train_data.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Image", "Label"])
#     writer.writerows(train_data)


def infer_stress(label):
    stress_labels = [2, 3, 5, 6]
    return 1 if label in stress_labels else 0

def get_image_path(i):
    if i < 10:
        return f"{rf_test}000{i}_aligned.jpg"
    elif 10 <= i < 100:
        return f"{rf_test}00{i}_aligned.jpg"
    elif 100 <= i < 1000:
        return f"{rf_test}0{i}_aligned.jpg"
    else:  
        return f"{rf_test}{i}_aligned.jpg"
    # else:
    #     return f"{rf_test}{i}_aligned.jpg"


df = pd.read_csv(filepath_test)
#df['Stress?'] = df['Label'].apply(infer_stress)
new_filenames = [get_image_path(i) for i in range(1, len(df['Images']) + 1)]
print(df.head())
# df['Images'] = new_filenames
df.to_csv('testdata_basic.csv')