from image_preprocessing import *
import cv2
from tensorflow.keras.models import load_model

# Load and process the image
def main():
    """
    Main function to load a sample Sudoku image, detect the grid, and display it.

    This function calls `detect_grid` to extract the Sudoku grid from an input image,
    then displays the detected grid using OpenCV.

    Example:
    --------
    >>> main()
    """
    image_path = '../data/test_images/sample_sudoku.jpg'
    grid_image = detect_grid(image_path)

    # Extract and preprocess the cells
    cells = extract_cells(grid_image)
    
    # Load Model
    model = load_model('../models/mnist_digit_recognizer.h5')

    # Predict digits in each cell
    predicted_digits = predict_digits(cells, model)

    # Reshape into 9x9 sudoku grid
    sudoku_grid = np.array(predict_digits).reshape(9, 9)

    print("Sudoku Grid:")
    print(sudoku_grid)

if __name__ == "__main__":
    main()
