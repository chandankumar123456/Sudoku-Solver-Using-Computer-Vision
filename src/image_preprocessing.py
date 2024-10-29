"""
This Script will:
    1. Read and grayscale the image
    2. Detect the largest square(the Sudoku grid) using contours
    3. Apply a perspective transformation to get a top-down view.
"""
import cv2
import numpy as np

def detect_grid(image_path):
    """
    Detects the Sudoku grid in an image and returns a top-down view of the grid.

    Parameters:
    -----------
    image_path : str
        Path to the Sudoku image file.

    Returns:
    --------
    warped : np.ndarray
        A transformed, top-down view of the detected Sudoku grid as an image array.
        
    Raises:
    -------
    ValueError:
        If a suitable square contour is not found in the image.
        
    Example:
    --------
    >>> grid_image = detect_grid('data/test_images/sample_sudoku.jpg')
    >>> cv2.imshow('Detected Grid', grid_image)
    >>> cv2.waitKey(0)
    """
    # Load the image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur and edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find Contours and find the area
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse=True)

    # Find the largest square Contour
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            target_contour = approx
            break
        else:
            raise ValueError("Sudoku grid not found in the image.")
        
    
    # Transform the perspective to get a top-down view
    pts = np.float32([target_contour[i][0] for i in range(4)])
    rect = np.array([pts[0], pts[1], pts[2], pts[3]], dtype = 'float32')

    # Compute the perspective transform matrix and apply it
    side = 450
    dst = np.array([[0, 0], [side - 1, 0], [side -1, side - 1], [0, side - 1]], dtype = "float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (side, side))

    return warped

def extract_cells(grid_image):
    """
    Splits the Sudoku grid image into individual cells for digit recognition.

    Parameters:
    -----------
    grid_image : np.ndarray
        A transformed top-down view of the Sudoku grid.

    Returns:
    --------
    List of np.ndarray
        A list of 81 images, each representing an individual cell in the grid.
    """
    cells = []
    side = grid_image.shape[0] // 9

    for i in range(9):
        for j in range(9):
            # Extract the cell from the grid_image by slicing out a sub-image
            # i * side : (i + 1) * side determines the row range for this cell
            # j * side : (j + 1) * side determines the column range for this cell
            cell = grid_image[i * side: (i + 1 ) * side, j * side: (j + 1) * side]
            cells.append(cell)
    
    return cells

def preprocess_cells(cell):
    """
    Prepares a cell image for digit prediction by resizing and normalizing.

    Parameters:
    -----------
    cell : np.ndarray
        The grayscale image of a single cell from the Sudoku grid.

    Returns:
    --------
    np.ndarray
        The preprocessed cell image ready for prediction (28x28 and normalized).
    """
    # Convert to grayscale if the image has color channels
    if cell.ndim == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    # Resize the cell to 28x28 pixels to match the model input shape
    resized_cell = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize pixel values 
    normalized_cell = resized_cell/255.0

    # Reshape to (28, 28, 1) for the model, which expects grayscale images with 1 channel
    prepared_cell = normalized_cell.reshape(28, 28, 1)

    return prepared_cell

def predict_digits(cells, model):
    """
    Predicts the digit in each cell using the trained model.

    Parameters:
    -----------
    cells : List of np.ndarray
        List of preprocessed cell images ready for prediction.

    model : keras.Model
        The trained digit recognition model.

    Returns:
    --------
    List of int
        A list of predicted digits for each cell in the Sudoku grid.
        Empty cells (no digit) should return 0.
    """
    predictions = []

    for cell in cells:
        # Preprocess the cell image for the model
        processed_cell = preprocess_cells(cell)
        processed_cell = np.expand_dims(processed_cell, axis=0)
        # Predict the digit 
        prediction = model.predict(processed_cell)

        # Get the digit
        predicted_digit = np.argmax(prediction, axis = -1)[0]

        # Append the digit to the predictions list
        predictions.append(predicted_digit)

    # Verify that we have exactly 81 predictions
    print(f"Number of predictions made: {len(predictions)}")  # Should be 81

    return predictions