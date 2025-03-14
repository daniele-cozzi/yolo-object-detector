import cv2
import os
from ultralytics import YOLO
import random

# Constants
MODEL_PATH = "yolo11n.pt"
INPUT_FOLDER = "images"
OUTPUT_FOLDER = "processed_images"

# Load the YOLO pre-trained model
model = YOLO(MODEL_PATH)

# Dictionary to store colors for each class. E.g. "person" -> (255, 0, 0)
class_colors = {}

def get_random_color():
    """
    Generates a very bright color in RGB format.
    :return: A random bright color as an RGB tuple. E.g. (255, 0, 0) for Red
    """
    color_options = [
        (255, 0, 0),  # Bright Red
        (0, 255, 0),  # Bright Green
        (0, 0, 255),  # Bright Blue
        (255, 255, 0),  # Bright Yellow
        (0, 255, 255),  # Bright Cyan
        (255, 0, 255),  # Bright Magenta
        (255, 165, 0),  # Bright Orange
        (255, 20, 147),  # Neon Pink
        (57, 255, 20),  # Neon Green
        (0, 191, 255)  # Electric Blue
    ]

    return random.choice(color_options)

def detect_objects_in_frame(frame):
    """
    Detect objects in a single frame using YOLO and draw bounding boxes and labels.

    :param frame: Image frame (numpy array)
    :return: Frame with bounding boxes and labels
    """
    # Perform YOLO on the frame and get a list of detected objects
    results = model(frame)

    # Iterate over each result (object)
    # result: detected object in the frame. It contains bounding boxes and names of objects
    for result in results:
        # Iterate over each bounding box
        # box: bounding box of an object. It contains coordinates, confidence score and class label
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box coordinates
            conf = float(box.conf[0]) # Confidence score (can be set a threshold)
            label = result.names[int(box.cls[0])] # Object label

            # Get color for the class, generate a new color if not found
            if label not in class_colors:
                class_colors[label] = get_random_color()
            color = class_colors[label]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def object_detection_webcam():
    """
    Perform real-time object detection using the webcam.
    Press 'q' to exit.
    """
    # Open the default webcam (index 0)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    while cap.isOpened():
        ret, frame = cap.read() # Read a frame
        if not ret:
            break

        # Process frame and display results
        frame = detect_objects_in_frame(frame)
        cv2.imshow("YOLO Object Detection (Webcam)", frame)

        # 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

def object_detection_on_images():
    """
    Perform object detection on all images in a specified folder and save the results.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Get all image files in the input folder (jpg, png, jpeg)
    images = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Iterate over each file in the input folder
    for image in images:
        # Read the image
        img_path = os.path.join(INPUT_FOLDER, image)
        img = cv2.imread(img_path)

        if img is not None:
            print(f"\nProcessing: {image}...")

            # Process the image and detect objects
            processed_img = detect_objects_in_frame(img)

            # Save the processed image
            output_path = os.path.join(OUTPUT_FOLDER, image)
            cv2.imwrite(output_path, processed_img)
            print(f"\nProcessed: {image} -> Saved to {OUTPUT_FOLDER}")

    print("\nAll images have been processed.")

def get_image_pairs():
    """
    Retrieve pairs of original and detected images.
    """
    # Array to store image pairs (original, detected)
    image_pairs = []

    # Get all image files in the input folder (jpg, png, jpeg)
    images = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Iterate over each file in the input folder and add to the image pairs array
    for image in images:
        original_path = os.path.join(INPUT_FOLDER, image)
        processed_path = os.path.join(OUTPUT_FOLDER, image)

        if os.path.exists(processed_path):
            original = cv2.imread(original_path)
            processed = cv2.imread(processed_path)
            if original is not None and processed is not None:
                image_pairs.append((original, processed))

    return image_pairs

def display_images():
    """
    Display images in pairs (original vs detected) with navigation.
    """
    image_pairs = get_image_pairs()

    if not image_pairs:
        print("No images found or processed.")
        return

    index = 0
    total_pairs = len(image_pairs)

    while True:
        original, processed = image_pairs[index]

        # Combine original and processed images side by side with a border
        border = cv2.copyMakeBorder(processed, 0, 0, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        combined = cv2.hconcat([original, border])

        cv2.imshow("Object Detection Viewer", combined)
        key = cv2.waitKeyEx(0)

        if key == 27: # ESC to exit
            break
        elif key == 2424832: # Left arrow key
            index = (index - 1) % total_pairs
        elif key == 2555904: # Right arrow key
            index = (index + 1) % total_pairs

    cv2.destroyAllWindows()

def main():
    """
    Main function to choose between real-time webcam detection or image processing.
    """
    while True:
        print("\nChoose an option:")
        print("1 - Real-time object detection (Webcam)")
        print("2 - Object detection on images from a folder")
        print("3 - Exit")

        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == "1":
            object_detection_webcam()
        elif choice == "2":
            object_detection_on_images()
            display_images()
        elif choice == "3":
            print("Exiting program.")
            break
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()