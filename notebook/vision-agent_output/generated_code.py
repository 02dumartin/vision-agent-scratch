import os
import numpy as np
from vision_agent.tools import *
from vision_agent.tools.planner_tools import judge_od_results
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool

from vision_agent.tools import load_image, countgd_object_detection, overlay_bounding_boxes, save_image
from typing import Dict, Any, List


def detect_and_count_objects(image_path: str, output_path: str = "detected_objects.png") -> Dict[str, Any]:
    """
    Detects and counts objects (tomatoes) in an image, visualizes the detections,
    and saves the result.
    
    This function uses object detection to identify and count all tomato instances
    in the provided image. It overlays bounding boxes on the detected objects and
    saves the visualization to a file.
    
    Parameters:
        image_path (str): The path to the input image file
        output_path (str): The path where the visualization will be saved. 
                          Defaults to "detected_objects.png"
    
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'count': The total number of objects detected
            - 'detections': List of detection dictionaries with 'score', 'label', and 'bbox'
            - 'output_image_path': Path to the saved visualization
    
    Example:
        >>> result = detect_and_count_objects("image.png")
        >>> print(f"Detected {result['count']} objects")
        Detected 4 objects
    """
    # Load the image
    image = load_image(image_path)
    
    # Detect tomatoes in the image using countgd_object_detection
    detections = countgd_object_detection("tomato", image)
    
    # Count the number of detections
    count = len(detections)
    
    # Visualize the detections with bounding boxes
    image_with_boxes = overlay_bounding_boxes(image, detections)
    
    # Save the visualization
    save_image(image_with_boxes, output_path)
    
    # Print results
    print(f"Total number of objects detected: {count}")
    for i, detection in enumerate(detections):
        print(f"Object {i+1}:")
        print(f"  Label: {detection['label']}")
        print(f"  Confidence: {detection['score']:.2f}")
        print(f"  Bounding box: {detection['bbox']}")
    print(f"\nVisualization saved to '{output_path}'")
    
    # Return the results
    return {
        'count': count,
        'detections': detections,
        'output_image_path': output_path
    }

import os
import numpy as np
from vision_agent.tools import *
from vision_agent.tools.planner_tools import judge_od_results
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool

from vision_agent.tools import load_image, countgd_object_detection, overlay_bounding_boxes, save_image
from typing import Dict, Any, List


def detect_and_count_objects(image_path: str, output_path: str = "detected_objects.png") -> Dict[str, Any]:
    """
    Detects and counts objects (tomatoes) in an image, visualizes the detections,
    and saves the result.
    
    This function uses object detection to identify and count all tomato instances
    in the provided image. It overlays bounding boxes on the detected objects and
    saves the visualization to a file.
    
    Parameters:
        image_path (str): The path to the input image file
        output_path (str): The path where the visualization will be saved. 
                          Defaults to "detected_objects.png"
    
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'count': The total number of objects detected
            - 'detections': List of detection dictionaries with 'score', 'label', and 'bbox'
            - 'output_image_path': Path to the saved visualization
    
    Example:
        >>> result = detect_and_count_objects("image.png")
        >>> print(f"Detected {result['count']} objects")
        Detected 4 objects
    """
    # Load the image
    image = load_image(image_path)
    
    # Detect tomatoes in the image using countgd_object_detection
    detections = countgd_object_detection("tomato", image)
    
    # Count the number of detections
    count = len(detections)
    
    # Visualize the detections with bounding boxes
    image_with_boxes = overlay_bounding_boxes(image, detections)
    
    # Save the visualization
    save_image(image_with_boxes, output_path)
    
    # Print results
    print(f"Total number of objects detected: {count}")
    for i, detection in enumerate(detections):
        print(f"Object {i+1}:")
        print(f"  Label: {detection['label']}")
        print(f"  Confidence: {detection['score']:.2f}")
        print(f"  Bounding box: {detection['bbox']}")
    print(f"\nVisualization saved to '{output_path}'")
    
    # Return the results
    return {
        'count': count,
        'detections': detections,
        'output_image_path': output_path
    }


def test_detect_and_count_objects():
    """
    Test case for detect_and_count_objects function.
    
    This test verifies:
    1. The function runs successfully with the provided image
    2. The return value is a dictionary with expected keys
    3. The 'count' field is a non-negative integer
    4. The 'detections' field is a list
    5. Each detection has the required fields: 'score', 'label', and 'bbox'
    6. The output image path matches the expected path
    """
    # Test with the provided image file
    image_path = "image.png"
    output_path = "detected_objects.png"
    
    # Call the function with the provided image
    result = detect_and_count_objects(image_path, output_path)
    
    # Print the result for verification
    print("\n=== Test Results ===")
    print(f"Result type: {type(result)}")
    print(f"Result keys: {result.keys()}")
    print(f"Object count: {result['count']}")
    print(f"Number of detections: {len(result['detections'])}")
    print(f"Output image path: {result['output_image_path']}")
    
    # Assert the output structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert 'count' in result, "Result should contain 'count' key"
    assert 'detections' in result, "Result should contain 'detections' key"
    assert 'output_image_path' in result, "Result should contain 'output_image_path' key"
    
    # Assert the data types
    assert isinstance(result['count'], int), "'count' should be an integer"
    assert result['count'] >= 0, "'count' should be non-negative"
    assert isinstance(result['detections'], list), "'detections' should be a list"
    assert isinstance(result['output_image_path'], str), "'output_image_path' should be a string"
    
    # Assert the output path matches
    assert result['output_image_path'] == output_path, "Output path should match the specified path"
    
    # If there are detections, verify their structure
    if result['count'] > 0:
        for detection in result['detections']:
            assert isinstance(detection, dict), "Each detection should be a dictionary"
            assert 'score' in detection, "Detection should contain 'score'"
            assert 'label' in detection, "Detection should contain 'label'"
            assert 'bbox' in detection, "Detection should contain 'bbox'"
            assert isinstance(detection['score'], (int, float)), "'score' should be numeric"
            assert isinstance(detection['label'], str), "'label' should be a string"
            assert isinstance(detection['bbox'], list), "'bbox' should be a list"
    
    print("\n=== All assertions passed ===")
    
    return result


# Run the test
test_detect_and_count_objects()
