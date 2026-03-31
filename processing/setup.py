import tensorflow as tf
import os
import json

# Fetches the TFRecord dataset from the data folder and returns a TFRecordDataset object.
def fetch_tfrecords():
    return tf.data.TFRecordDataset(["./data/objects.tfrecord"])

# Extracts the encoded images and their filenames from TFRecordDataset, and saves the images to the data/img folder.
# Returns a map of image filenames to their corresponding labels
def tfrecords_to_images(raw_dataset):
    OUTPUT_DIR = "./data/img"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def _parse_image_function(example_proto):
        feature_description = {
            'image/encoded':  tf.io.FixedLenFeature([], tf.string),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        }
        return tf.io.parse_single_example(example_proto, feature_description)

    image_to_label = {}
    for i, raw_record in enumerate(raw_dataset):
        parsed = _parse_image_function(raw_record)

        # Get filename, fall back to index if empty
        filename = parsed['image/filename'].numpy().decode('utf-8')
        filename = os.path.basename(filename) if filename else f"image_{i:05d}.jpg"

        # Skip images with no labels (based on data exploration, there are only )
        if parsed['image/object/class/label'].values.shape[0] == 0:
            continue

        # Take the first label. While some images have multiple labels, all of them are duplicates.
        label_ids = tf.sparse.to_dense(parsed['image/object/class/label']).numpy().tolist()[0]

        image_to_label[filename] = label_ids  # Placeholder for label, to be filled in later

        # Save the image to the output directory
        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, 'wb') as f:
            f.write(parsed['image/encoded'].numpy())

        # Print progress 
        if (i + 1) % 100 == 0:
            print(f"Saved {i + 1} images...")
    
    # Save the mapping of image filenames to labels for later use
    with open("./data/chords/labels.json", "w") as f:
        json.dump(image_to_label, f, indent=2)

# Runs the data processing pipeline: fetches the TFRecords and extracts and saves the images
def main():
    raw_dataset = fetch_tfrecords()
    tfrecords_to_images(raw_dataset)

main()