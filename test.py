import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import csv

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Load the saved test data
X_test = np.load('X_test_v2.npy')
y_test = np.load('y_test_v2.npy')

# Load the saved model
model = load_model('sign_language_detection_model_v2.h5')

# Make predictions
y_pred = model.predict(X_test)

# Convert predictions and true values from one-hot encoded format to label format
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Evaluate the model
accuracy = accuracy_score(y_test_labels, y_pred_labels)
classification_rep = classification_report(y_test_labels, y_pred_labels, target_names=actions)
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)


# Save evaluation metrics to a text file
with open('evaluation_metrics_v2.txt', 'w') as file:
    file.write(f"Test Accuracy: {accuracy}\n\n")
    file.write("Classification Report:\n")
    file.write(classification_rep + '\n\n')
    file.write("Confusion Matrix:\n")
    file.write(str(conf_matrix))

