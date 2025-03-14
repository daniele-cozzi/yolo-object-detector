# **YOLO Object Detection - Real-time & Image Processing**

This Python project is a **case study** developed and presented during a **lesson I conducted as a tutor at an ITIS school** as part of a **PCTO (Percorsi per le Competenze Trasversali e l'Orientamento)** program. The objective of this lesson was to introduce students to **computer vision**, **object detection**, and practical applications of **YOLO (You Only Look Once)** in real-world scenarios.

**Presentation link:** https://pitch.com/v/computer-vision-evqtzw

## **Features**

- **Real-time Object Detection** using a webcam.
- **Batch Processing of Images** from a specified folder.
- **Color-coded Bounding Boxes** for detected objects.
- **Navigation System** to compare original and processed images.

## **How It Works**

1. **Real-time Detection:** Opens the webcam, detects objects, and overlays bounding boxes.
2. **Image Processing:** Loads images from a folder, applies object detection, and saves results.
3. **Image Viewer:** Allows users to navigate through processed images.

## **Requirements**

- Python 3.10
- OpenCV (`cv2`)
- Ultralytics YOLO (`ultralytics`)
- A YOLO model file (`yolo11n.pt`)

## **Installation**

Before running the script, install the necessary dependencies:
```bash
pip install opencv-python ultralytics
```

## **Usage**

Run the script and choose between real-time detection or image processing:

```bash
python object_detector.py
```

## **Customization**

- Change `MODEL_PATH` to use a different YOLO model.
- Modify `INPUT_FOLDER` and `OUTPUT_FOLDER` for custom image paths.
- Adjust the color palette for bounding boxes.
