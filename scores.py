from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import accuracy_score
import face_recognition
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

lfw_pairs_test = fetch_lfw_pairs(subset='test', color=True) # 1000 instances, first half are the same person while second half are different
pairs = lfw_pairs_test.pairs # Number of pairs: 1000
labels = lfw_pairs_test.target # labels whether same person or different
target_names = lfw_pairs_test.target_names  

THRESHOLD = 0.6

predictions = []
confidences = []

for i, (pair, label) in enumerate(zip(pairs, labels)):
    img1, img2 = pair

    # print(f"Pair {i}: image1 shape: {img1.shape}, image2 shape: {img2.shape}")
    # print(f"Pair {i}: img1 dtype: {img1.dtype}, img2 dtype: {img2.dtype}")
    # Convert to RGB
    img1_uint = (img1 * 255).astype(np.uint8)
    img2_uint = (img2 * 255).astype(np.uint8)

    encodings1 = face_recognition.face_encodings(img1_uint)
    encodings2 = face_recognition.face_encodings(img2_uint)

    if len(encodings1) == 0:
        print(f"Image detection failed in image 1 of pair {i}")
        predictions.append(0)
        continue
    if len(encodings2) == 0:
        print(f"Detection failed in image 2 of pair {i}")
        predictions.append(0)
        continue
    
    # Calculate distance

    face_distances = face_recognition.face_distance([encodings1[0]], encodings2[0])
    confidence = 1 - face_distances

    
    confidences.append(confidence)
    prediction = 1 if face_distances < THRESHOLD else 0
    predictions.append(prediction)

accuracy = accuracy_score(labels, predictions)
print(accuracy)

def plot_confusion_matrix(predictions, true_labels):
    """Create confusion matrix for face verification"""
    
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Different Person', 'Same Person'],
                yticklabels=['Different Person', 'Same Person'])
    plt.title('Confusion Matrix - Face Recognition on LFW')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add accuracy text
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    plt.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.1%}', 
             transform=plt.gca().transAxes, ha='center')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(labels, predictions)