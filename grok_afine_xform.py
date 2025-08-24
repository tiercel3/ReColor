import numpy as np
from PIL import Image
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Transform image1 to match the lighting conditions of image2 using color standards.")
    parser.add_argument("C1", type=str, help="Path to CSV file with Nx3 RGB values for color standards in image1")
    parser.add_argument("C2", type=str, help="Path to CSV file with Nx3 RGB values for color standards in image2")
    parser.add_argument("image1", type=str, help="Path to input image1 (.jpg or .png)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output of intermediate calculations")
    return parser.parse_args()

def load_csv_matrix(file_path):
    """Load an Nx3 CSV file into a NumPy array, ensuring at least 5 rows."""
    try:
        matrix = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        if matrix.shape[1] != 3:
            raise ValueError(f"CSV file {file_path} must have exactly 3 columns (RGB values).")
        if matrix.shape[0] < 5:
            raise ValueError(f"CSV file {file_path} must contain at least 5 rows of color standards.")
        return matrix
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        exit(1)

def transform_image(image_array, T, b):
    """Apply affine transformation (T * p + b) to each pixel in the image array."""
    height, width, _ = image_array.shape
    # Reshape image for vectorized operation: (height*width, 3)
    pixels = image_array.reshape(-1, 3).astype(np.float32)
    # Apply transformation: T * p + b
    transformed_pixels = pixels @ T.T + b
    # Clip to valid RGB range [0, 255]
    transformed_pixels = np.clip(transformed_pixels, 0, 255)
    # Reshape back to image dimensions
    transformed_image = transformed_pixels.reshape(height, width, 3).astype(np.uint8)
    return transformed_image

def get_output_filename(input_filename):
    """Generate output filename by appending '_xfrm' to the root of the input filename."""
    base, _ = os.path.splitext(input_filename)
    return f"{base}_xfrm.png"

def main():
    args = parse_args()

    # Load input files
    C1 = load_csv_matrix(args.C1)
    C2 = load_csv_matrix(args.C2)
    
    # Ensure C1 and C2 have the same number of rows
    if C1.shape[0] != C2.shape[0]:
        print(f"Error: C1 has {C1.shape[0]} rows, but C2 has {C2.shape[0]} rows. They must match.")
        exit(1)

    # Load image1 using PIL
    try:
        image1 = Image.open(args.image1)
        if image1.mode != 'RGB':
            image1 = image1.convert('RGB')
        image_array = np.array(image1, dtype=np.float32)
    except Exception as e:
        print(f"Error: Could not load image {args.image1}. Ensure it's a valid .jpg or .png file. Error: {e}")
        exit(1)

    # Step 1: Compute transformation
    # X = C1^T (3xN), Y = C2^T (3xN)
    X = C1.T
    Y = C2.T
    # Z = [X; ones(1,N)] (4xN)
    Z = np.vstack([X, np.ones((1, C1.shape[0]))])
    # Solve W = Y * pinv(Z) (3x4)
    W = Y @ np.linalg.pinv(Z)
    # Extract T (3x3) and b (3x1)
    T = W[:, 0:3]
    b = W[:, 3]

    # Verbose output
    if args.verbose:
        print("Intermediate calculations:")
        print("X (C1^T):\n", X)
        print("Y (C2^T):\n", Y)
        print("Z ([X; ones]):\n", Z)
        print("W (Y * pinv(Z)):\n", W)
        print("T (linear transformation):\n", T)
        print("b (translation):\n", b)

    # Step 2: Apply transformation to image1
    transformed_array = transform_image(image_array, T, b)

    # Save output as PNG using PIL
    output_path = get_output_filename(args.image1)
    transformed_image = Image.fromarray(transformed_array, mode='RGB')
    transformed_image.save(output_path, format='PNG')
    print(f"Transformed image saved as {output_path}")

if __name__ == "__main__":
    main()
