import json
from pathlib import Path
from glob import glob
from collections import Counter
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision  

# Load images from a folder and return in PIL format and file path format
def load_images_from_folder(folder_path):
    image_files = glob(os.path.join(folder_path, "*.jpg"))
    images_list = [Image.open(file) for file in image_files]
    print(f"Loaded {len(images_list)} images.")

    return images_list, image_files

# Perform landmark detection on a list of images using Google MediaPipe hand landmarking  
def landmark_detection(image_paths, hand_landmarker_model='models/hand_landmarker.task'):
    # Source: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#python_example
    results = []
    for image_path in image_paths:
        base_options = python.BaseOptions(model_asset_path=hand_landmarker_model)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                            num_hands=1, min_hand_detection_confidence=0.3)
        detector = vision.HandLandmarker.create_from_options(options)

        image = mp.Image.create_from_file(image_path)
        detection_result = detector.detect(image)
        results.append(detection_result)
    return results


# Visualize the landmarking results on the original images to see how accurately they mapped coordinates 
# over our images 
def processing(landmark_results, image_paths):
    """Visualize hand landmarks on the original images."""
    # Hand connection indices (MediaPipe hand model defines these connections)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),      # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),    # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
        (5, 9), (9, 13), (13, 17),                # Connections between fingers
    ]
    
    fig, axes = plt.subplots(len(landmark_results), 1, figsize=(8, 4 * len(landmark_results)))
    if len(landmark_results) == 1:
        axes = [axes]
    
    for idx, (result, image_path) in enumerate(zip(landmark_results, image_paths)):
        # Load and display the image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        h, w = image_array.shape[:2]
        
        # Draw landmarks if detected
        if result.hand_landmarks:
            axes[idx].imshow(image_array)
            for hand_landmarks in result.hand_landmarks:
                # Draw connections first (so they appear behind the points)
                for start, end in HAND_CONNECTIONS:
                    start_pos = hand_landmarks[start]
                    end_pos = hand_landmarks[end]
                    x_coords = [start_pos.x * w, end_pos.x * w]
                    y_coords = [start_pos.y * h, end_pos.y * h]
                    axes[idx].plot(x_coords, y_coords, 'c-', linewidth=2)
                
                # Draw landmarks (points)
                for landmark in hand_landmarks:
                    x = landmark.x * w
                    y = landmark.y * h
                    axes[idx].plot(x, y, 'ro', markersize=6)
            
            axes[idx].set_title(f"Hand landmarks detected in {Path(image_path).name}")
        else:
            axes[idx].set_title(f"No hand landmarks detected in {Path(image_path).name}")
        
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def load_label_map(labels_file='data/chords/labels.json'):
    labels_file = Path(labels_file)
    if not labels_file.exists():
        raise FileNotFoundError(f"Label file not found: {labels_file}")
    with labels_file.open('r') as f:
        return json.load(f)

def load_chord_map(chords_file='data/chords/chords.json'):
    chords_file = Path(chords_file)
    if not chords_file.exists():
        raise FileNotFoundError(f"Label file not found: {chords_file}")
    with chords_file.open('r') as f:
        return json.load(f)

def print_class_imbalance(image_paths, landmark_results, labels_file='data/chords/labels.json'):
    label_map = load_label_map(labels_file)
    all_labels = [label_map.get(Path(path).name, 'UNKNOWN') for path in image_paths]
    detected_labels = [label_map.get(Path(path).name, 'UNKNOWN') for path, result in zip(image_paths, landmark_results) if result.hand_landmarks]

    def format_counts(counter):
        lines = []
        total = sum(counter.values())
        for label, count in counter.most_common():
            pct = count / total * 100 if total else 0.0
            lines.append(f"  {label}: {count} ({pct:.1f}%)")
        return '\n'.join(lines)

    print("\nClass imbalance for all input images:")
    print(format_counts(Counter(all_labels)))
    print("\nClass imbalance for images with detected landmarks:")
    print(format_counts(Counter(detected_labels)))


# Fetch images and image paths 
images, image_paths = load_images_from_folder('data/img')

# Perform landmark detection on the first 10 images
landmark_results = landmark_detection(image_paths[:10])
print(landmark_results)
all_results = []

for landmark_result in landmark_results:
    if landmark_result.hand_landmarks:
        coords = []
        for landmark in landmark_result.hand_landmarks[0]:  # first hand
            coords.extend([landmark.x, landmark.y, landmark.z])  # flatten
        all_results.append(coords)

# Convert to NumPy array
np_arr_results = np.array(all_results)

# Save to CSV
np.savetxt('landmark_results.csv', np_arr_results, delimiter=',')
print_class_imbalance(image_paths, landmark_results)
x = 0
for result in landmark_results:
    if result.hand_landmarks:
        x += 1
print(f"Landmarks detected in {x} out of {len(landmark_results)} images.")
# Landmarks detected in 477/3389 w/ 0.5 confidence threshold 
# By decreasing the confidence threshold 0.3 we got 635/3389 