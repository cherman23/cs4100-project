from PIL import Image
from glob import glob
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load an example image. Eventually this will be replaced with our actual data from GoogleDrive.
def load_images_from_folder(folder_path):
    # Use glob to find all PNG files in the specified folder
    image_files = glob(os.path.join(folder_path, "*.jpg"))
    
    # Load each image and store it in a list
    images_list = [Image.open(file) for file in image_files]
    print(f"Loaded {len(images_list)} images.")

    return images_list, image_files

def landmark_detection(image_paths):
    # Perform landmark detection on each image and store the results in a list
    results = []
    for image_path in image_paths:
        # STEP 2: Create an HandLandmarker object.
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                            num_hands=1)
        detector = vision.HandLandmarker.create_from_options(options)

        # STEP 3: Load the input image.
        image = mp.Image.create_from_file(image_path)

        # STEP 4: Detect hand landmarks from the input image.
        detection_result = detector.detect(image)
        results.append(detection_result)
    return results

def processing(landmark_results):
    # Process the landmark results as needed (e.g., extract coordinates, visualize, etc.)
    for result in landmark_results:
        # Example: Print the detected landmarks for each hand
        for hand_landmarks in result.hand_landmarks:
            print(hand_landmarks)

images, image_paths = load_images_from_folder('./img')
landmark_results = landmark_detection(image_paths)
print(landmark_results)
