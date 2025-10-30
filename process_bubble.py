#!/usr/bin/env python3
"""
Bubble Processing Module
========================

Processes detected text bubbles to prepare them for text replacement.
This module handles the cleaning and preparation of speech bubble regions
by removing existing text and creating a clean background.

Author: MangaTranslator Team
License: MIT
"""

import cv2
import numpy as np


def process_bubble(image):
    """
    Process a speech bubble by removing existing text and creating a clean background

    This function analyzes the bubble region, detects the bubble boundary,
    and fills the interior with white color to prepare for new text insertion.

    Args:
        image (numpy.ndarray): Input image containing the speech bubble (BGR format)

    Returns:
        tuple: A tuple containing:
            - image (numpy.ndarray): Processed image with white bubble interior
            - largest_contour (numpy.ndarray): Contour of the detected bubble boundary
    """
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to separate bubble from background
    # Threshold value 240 works well for typical manga bubbles
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour (assumed to be the speech bubble)
    if not contours:
        return image, None
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the bubble area
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)

    # Fill the bubble area with white color
    image[mask == 255] = (255, 255, 255)

    return image, largest_contour
