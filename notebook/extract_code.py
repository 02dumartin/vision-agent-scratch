import numpy as np
from PIL import Image
from vision_agent.tools.tools import agentic_object_detection
import cv2


def load_image(image_path: str) -> np.ndarray:
    """Load image from file path and return as numpy array."""
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")


def overlay_bounding_boxes(image: np.ndarray, bounding_boxes: list, labels: list = None) -> np.ndarray:
    """Overlay bounding boxes on the image."""
    overlay_image = image.copy()
    height, width = image.shape[:2]
    
    for i, bbox in enumerate(bounding_boxes):
        x1, y1, x2, y2 = bbox
        # Convert normalized coordinates to pixel coordinates
        x1_px = int(x1 * width)
        y1_px = int(y1 * height)
        x2_px = int(x2 * width)
        y2_px = int(y2 * height)
        
        # Draw bounding box
        cv2.rectangle(overlay_image, (x1_px, y1_px), (x2_px, y2_px), (0, 255, 0), 2)
        
        # Add label if provided
        if labels and i < len(labels):
            label_text = labels[i]
            cv2.putText(overlay_image, label_text, (x1_px, y1_px - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return overlay_image


def run(image_path: str) -> dict:
    """Main function to detect semi-ripe tomatoes and create summary report."""
    try:
        # Load the image
        image = load_image(image_path)
        height, width = image.shape[:2]
        
        # Use agentic object detection to find semi-ripe tomatoes
        detections = agentic_object_detection('semi-ripe tomato', image)
        
        # Extract and format bounding box coordinates
        bounding_boxes = []
        formatted_detections = []
        
        for i, detection in enumerate(detections):
            # Extract normalized coordinates
            bbox = detection['bbox']
            x1_norm, y1_norm, x2_norm, y2_norm = bbox
            
            # Convert to pixel coordinates
            x1_px = int(x1_norm * width)
            y1_px = int(y1_norm * height)
            x2_px = int(x2_norm * width)
            y2_px = int(y2_norm * height)
            
            bounding_boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
            
            formatted_detection = {
                'detection_id': i + 1,
                'label': detection.get('label', 'semi-ripe tomato'),
                'confidence': detection.get('score', 1.0),
                'normalized_bbox': [x1_norm, y1_norm, x2_norm, y2_norm],
                'pixel_bbox': [x1_px, y1_px, x2_px, y2_px],
                'bbox_width_px': x2_px - x1_px,
                'bbox_height_px': y2_px - y1_px,
                'center_x_px': (x1_px + x2_px) // 2,
                'center_y_px': (y1_px + y2_px) // 2
            }
            formatted_detections.append(formatted_detection)
        
        # Create visualization with bounding boxes
        labels = [f"semi-ripe tomato {i+1}" for i in range(len(bounding_boxes))]
        visualization = overlay_bounding_boxes(image, bounding_boxes, labels)
        
        # Create summary report
        summary_report = {
            'image_info': {
                'image_path': image_path,
                'image_dimensions': {
                    'width': width,
                    'height': height
                }
            },
            'detection_summary': {
                'total_semi_ripe_tomatoes': len(detections),
                'prompt_used': 'semi-ripe tomato'
            },
            'detections': formatted_detections,
            'visualization_created': True
        }
        
        # Print summary
        print(f"Semi-ripe Tomato Detection Report")
        print(f"==================================")
        print(f"Image: {image_path}")
        print(f"Image dimensions: {width} x {height}")
        print(f"Total semi-ripe tomatoes detected: {len(detections)}")
        print(f"\nDetection details:")
        
        for detection in formatted_detections:
            print(f"  Tomato {detection['detection_id']}:")
            print(f"    - Confidence: {detection['confidence']:.3f}")
            print(f"    - Pixel bbox: {detection['pixel_bbox']}")
            print(f"    - Center: ({detection['center_x_px']}, {detection['center_y_px']})")
            print(f"    - Size: {detection['bbox_width_px']} x {detection['bbox_height_px']} px")
        
        return summary_report
        
    except Exception as e:
        error_report = {
            'error': str(e),
            'image_path': image_path,
            'detection_summary': {
                'total_semi_ripe_tomatoes': 0,
                'prompt_used': 'semi-ripe tomato'
            },
            'detections': []
        }
        print(f"Error processing image: {e}")
        return error_report


if __name__ == "__main__":
    result = run("data/tomato_farm.jpg")