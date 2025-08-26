#!/usr/bin/env python3
"""
Bubble Detection Module - Optimized
===================================

Uses YOLO (You Only Look Once) model to detect text bubbles in manga/comic images
with advanced filtering to prevent duplicates and overlapping detections.

Features:
- Confidence threshold filtering
- Non-Maximum Suppression (NMS) 
- Overlap filtering
- Size-based filtering

Author: MangaTranslator Team
License: MIT
"""

import torch.serialization
from ultralytics import YOLO
import numpy as np


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1, box2: [x1, y1, x2, y2, confidence, class_id]
        
    Returns:
        float: IoU value between 0 and 1
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    # Calculate intersection area
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def filter_overlapping_bubbles(bubbles, iou_threshold=0.3):
    """
    Remove overlapping bubbles using IoU threshold
    
    Args:
        bubbles: List of bubble detections
        iou_threshold: IoU threshold for considering bubbles as overlapping
        
    Returns:
        list: Filtered list without overlapping bubbles
    """
    if len(bubbles) <= 1:
        return bubbles
    
    # Sort by confidence score (descending)
    bubbles = sorted(bubbles, key=lambda x: x[4], reverse=True)
    
    filtered_bubbles = []
    
    for i, bubble in enumerate(bubbles):
        # Check if this bubble overlaps significantly with any already accepted bubble
        is_duplicate = False
        
        for accepted_bubble in filtered_bubbles:
            iou = calculate_iou(bubble, accepted_bubble)
            if iou > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_bubbles.append(bubble)
    
    return filtered_bubbles


def filter_by_size(bubbles, min_width=20, min_height=20, max_width_ratio=0.8, max_height_ratio=0.8, image_width=None, image_height=None):
    """
    Filter bubbles by size constraints
    
    Args:
        bubbles: List of bubble detections
        min_width, min_height: Minimum bubble dimensions
        max_width_ratio, max_height_ratio: Maximum size relative to image
        image_width, image_height: Image dimensions for ratio calculation
        
    Returns:
        list: Size-filtered bubbles
    """
    filtered_bubbles = []
    
    for bubble in bubbles:
        x1, y1, x2, y2 = bubble[:4]
        width = x2 - x1
        height = y2 - y1
        
        # Check minimum size
        if width < min_width or height < min_height:
            continue
        
        # Check maximum size ratio (if image dimensions provided)
        if image_width and image_height:
            width_ratio = width / image_width
            height_ratio = height / image_height
            
            if width_ratio > max_width_ratio or height_ratio > max_height_ratio:
                continue
        
        filtered_bubbles.append(bubble)
    
    return filtered_bubbles


def detect_bubbles(model_path, image_path, conf_threshold=0.25, iou_threshold=0.3, enable_nms=True):
    """
    Detect text bubbles in manga/comic images using YOLOv8 model with advanced filtering
    
    This function loads a pre-trained YOLO model and uses it to identify
    text bubble regions with duplicate removal and overlap filtering.
    
    Args:
        model_path (str): Path to the YOLO model file (.pt format)
        image_path (str): Path to the input image or PIL Image object
        conf_threshold (float): Confidence threshold for detections (0.0-1.0)
        iou_threshold (float): IoU threshold for overlap filtering (0.0-1.0)
        enable_nms (bool): Enable Non-Maximum Suppression in YOLO
        
    Returns:
        list: List of detected bubbles with format:
              [x1, y1, x2, y2, confidence_score, class_id]
              where (x1,y1) is top-left corner and (x2,y2) is bottom-right corner
              
    Note:
        - Lower conf_threshold = more detections (but more false positives)
        - Lower iou_threshold = more aggressive overlap removal
        - Results are filtered and sorted by confidence
    """
    # Load YOLO model with safe globals for security
    with torch.serialization.safe_globals([YOLO]):
        model = YOLO(model_path)

    # Configure model parameters
    model.overrides['conf'] = conf_threshold  # Confidence threshold
    model.overrides['iou'] = 0.7 if enable_nms else 1.0  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # Class-agnostic NMS
    model.overrides['max_det'] = 100  # Maximum detections per image

    # Run detection on the image
    results = model(image_path)[0]

    # Extract bounding box data
    if results.boxes is None or len(results.boxes) == 0:
        print("⚠️ No bubbles detected")
        return []
    
    bubbles = results.boxes.data.tolist()
    original_count = len(bubbles)
    
    print(f"🔍 Initial detections: {original_count}")

    # Apply confidence filtering (additional safety check)
    bubbles = [bubble for bubble in bubbles if bubble[4] >= conf_threshold]
    conf_filtered_count = len(bubbles)
    
    if conf_filtered_count < original_count:
        print(f"🎯 After confidence filter ({conf_threshold}): {conf_filtered_count}")

    # Get image dimensions for size filtering
    image_height, image_width = None, None
    if hasattr(results, 'orig_shape'):
        image_height, image_width = results.orig_shape
    
    # Apply size filtering
    bubbles = filter_by_size(
        bubbles, 
        min_width=15, 
        min_height=15,
        max_width_ratio=0.9, 
        max_height_ratio=0.9,
        image_width=image_width,
        image_height=image_height
    )
    size_filtered_count = len(bubbles)
    
    if size_filtered_count < conf_filtered_count:
        print(f"📏 After size filter: {size_filtered_count}")

    # Apply overlap filtering (our custom IoU-based filter)
    bubbles = filter_overlapping_bubbles(bubbles, iou_threshold)
    final_count = len(bubbles)
    
    if final_count < size_filtered_count:
        print(f"🎯 After overlap filter (IoU {iou_threshold}): {final_count}")

    # Sort final results by Y coordinate (top to bottom) then by confidence
    bubbles = sorted(bubbles, key=lambda x: (x[1], -x[4]))
    
    # Print filtering summary
    removed_count = original_count - final_count
    if removed_count > 0:
        print(f"✨ Removed {removed_count} duplicate/overlapping bubbles ({removed_count/original_count*100:.1f}%)")
    
    return bubbles
