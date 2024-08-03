import json

# Read the labels from the labels.txt file
with open('C:/Users/khatr/OneDrive/Desktop/Study/Internship Tings/Food101/meta/labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Create a dictionary with indices as keys and labels as values
labels_dict = {i: label for i, label in enumerate(labels)}

# Save the dictionary as a JSON file
with open('classes.json', 'w') as json_file:
    json.dump(labels_dict, json_file)