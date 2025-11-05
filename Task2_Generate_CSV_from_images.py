import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

DATASET_PATH = "/dtu/datasets1/02516/PH2_Dataset_images/"
OUTPUT_PATH = os.path.expanduser("~/Project3/image-segmentation/Task2_dataset")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Get list of all IMD folders
folders = sorted([f for f in os.listdir(DATASET_PATH) if f.startswith("IMD")])

def sample_points_along_contour(contour, num_points=6, jitter=2):
    # Function to take a contour and get evenly spaced points along the contor and add some randomness to them
    # Flatten contour: Nx2
    contour = contour.reshape(-1, 2)
    
    # Compute cumulative distances along contour
    diffs = np.diff(contour, axis=0, append=[contour[0]])
    dists = np.sqrt((diffs[:,0])**2 + (diffs[:,1])**2)
    cumdist = np.cumsum(dists)
    cumdist /= cumdist[-1]  # normalize 0→1

    # Sample points evenly along normalized distance
    sample_positions = np.linspace(0, 1, num_points, endpoint=False)
    sampled_points = []
    for pos in sample_positions:
        idx = np.searchsorted(cumdist, pos)
        x, y = contour[idx]
        # Add small random jitter
        x += random.randint(-jitter, jitter)
        y += random.randint(-jitter, jitter)
        sampled_points.append([x, y])
    
    return sampled_points


def visualize_points(image: np.ndarray, points: np.ndarray, output_path: str, filename: str = "sample_points_overlay.png"):
    # function to visualize points on the image
    vis_img = image.copy()

    for (x, y, c) in points:
        color = (0, 0, 255) if c == 1 else (0, 255, 0)
        cv2.circle(vis_img, (int(x), int(y)), 4, color, -1)

    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, filename)
    cv2.imwrite(save_path, vis_img)
    print(f"Visualization saved at: {save_path}")

def extract_points_from_image(image: np.ndarray, mask: np.ndarray):
    #functoin to take the image and the lesion mask,
    #apply erosion for the lesion and dilation for the non-lesion region
    #find contors
    #find evenly spaced points along the contor and add randomness
    #visualize some random images
    #store the points in a csv alon with lables 1-> lesion, 0->nonlesion

    erosion_iters = 50
    # --- Step 1: Prepare binary mask ---
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)

    # --- Step 2: Lesion region ---
    lesion_eroded = cv2.erode(binary_mask, kernel, iterations=erosion_iters)

    # Get lesion contour and simplify
    contours, _ = cv2.findContours(lesion_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lesion_points = []

    if contours:
        contour = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(contour, epsilon=2.0, closed=True)
        approx = approx.reshape(-1, 2)

        sampled = sample_points_along_contour(approx, num_points=6, jitter=2)
        for x, y in sampled:
            lesion_points.append([x, y, 1])

    # --- Step 3: Non-lesion region ---
    # inverted = cv2.bitwise_not(binary_mask)
    inverted = binary_mask.copy()

    nonlesion_eroded = cv2.dilate(inverted, kernel, iterations=erosion_iters)

    contours, _ = cv2.findContours(nonlesion_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nonlesion_points = []

    if contours:
        contour = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(contour, epsilon=2.0, closed=True)
        approx = approx.reshape(-1, 2)

        sampled = sample_points_along_contour(approx, num_points=6, jitter=2)
        for x, y in sampled:
            lesion_points.append([x, y, 0])

    # --- Step 4: Combine and return ---
    all_points = np.array(lesion_points + nonlesion_points, dtype=int)

    return all_points


for folder in tqdm(folders, desc="Processing images"):
    try:
        # Construct paths
        image_path = os.path.join(DATASET_PATH, folder, f"{folder}_Dermoscopic_Image", f"{folder}.bmp")
        mask_path = os.path.join(DATASET_PATH, folder, f"{folder}_lesion", f"{folder}_lesion.bmp")

        # Read images
        image = cv2.imread(image_path)                      # Color image (BGR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        if image is None or mask is None:
            print(f"Skipping {folder} (missing files)")
            continue

        # Convert lesion mask to binary (0 or 1)
        _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        # Get coordinates
        height, width = binary_mask.shape
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        # Flatten everything
        data = extract_points_from_image(image, mask)

        # Save to CSV
        csv_name = f"{folder}.csv"
        csv_path = os.path.join(OUTPUT_PATH, csv_name)

        df = pd.DataFrame(data, columns=["x", "y", "lesion"])
        df.to_csv(csv_path, index=False)
        print("CSV File created")

        # Randomly visualize only one image from all
        if random.random() < 0.05:  # ~5% chance, effectively one or a few total
            visualize_points(image, data, OUTPUT_PATH, filename=f"{folder}_points_overlay.png")

    except Exception as e:
        print(f"Error processing {folder}: {e}")

print("\nAll images processed. CSV files saved in:", OUTPUT_PATH)
