import cv2
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
import webcolors
import os
import warnings
import requests
from io import BytesIO

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

# Load CSS3 color names and their corresponding RGB values
color_names_to_hex = webcolors.CSS3_NAMES_TO_HEX
color_rgbs = [webcolors.hex_to_rgb(hex_val) for hex_val in color_names_to_hex.values()]
color_names = list(color_names_to_hex.keys())

def rgb_to_name(rgb_triplet):
    min_distance = float('inf')
    closest_name = None
    
    # Calculate Euclidean distance between the RGB value and each predefined color
    for name, rgb in zip(color_names, color_rgbs):
        distance = np.linalg.norm(np.array(rgb_triplet) - np.array(rgb))
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    
    return closest_name

def find_dominant_colors(image_path, k=8, resize_factor=0.1, num_runs=3):
    try:
        # Open image using PIL
        image = Image.open(image_path)
        
        # Resize image to a fixed size
        target_width, target_height = 300, 300  # Fixed size for consistency
        image = image.resize((target_width, target_height))
        
        # Convert image to RGB mode
        image = image.convert('RGB')
        
        # Convert image to numpy array
        image_array = np.array(image)
        
        # Reshape the image to be a list of pixels
        pixels = image_array.reshape(-1, 3)
        
        best_dominant_colors = None
        
        for _ in range(num_runs):
            # Perform KMeans clustering with KMeans++ initialization
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
            kmeans.fit(pixels)
            
            # Get cluster centers
            centers = np.uint8(kmeans.cluster_centers_)
                
            # Find count of pixels for each centroid
            labels = kmeans.labels_
            counts = np.bincount(labels)
                
            # Get the top k dominant colors
            sorted_centers = centers[np.argsort(-counts)]
            sorted_counts = np.sort(counts)[::-1]
                
            # Calculate percentages
            total_pixels = np.sum(counts)
            percentages = sorted_counts / total_pixels * 100
                
            dominant_colors = list(zip(sorted_centers, percentages, [rgb_to_name(color) for color in sorted_centers]))
                
            # Ensure the highest dominant color is included in the list
            dominant_color_index = np.argmax(counts)
            highest_dominant_color = centers[dominant_color_index]
            highest_dominant_color_percentage = 100 * counts[dominant_color_index] / total_pixels
            if not any(np.array_equal(color, highest_dominant_color) for color, _, _ in dominant_colors):
                color_name = rgb_to_name(highest_dominant_color)
                dominant_colors.append((highest_dominant_color, highest_dominant_color_percentage, color_name))
                
            # Sort dominant colors based on their percentages
            dominant_colors.sort(key=lambda x: x[1], reverse=True)
                
            # Exclude repeated color names and the highest dominant color
            unique_color_names = set()
            unique_dominant_colors = []
            for color in dominant_colors[:5]:  # Display top 5 colors
                if color[2] not in unique_color_names:
                    unique_color_names.add(color[2])
                    unique_dominant_colors.append(color)
                
            best_dominant_colors = unique_dominant_colors
        
        return best_dominant_colors
    except Exception as e:
        print("Error:", e)
        return None

def mark_highest_dominant_color(image_path, k=64):
    try:
        # Open image using PIL
        image = Image.open(image_path)
        
        # Convert image to RGBA mode
        image = image.convert('RGBA')
        
        # Convert image to numpy array
        image_array = np.array(image)
        
        # Reshape the image to be a list of pixels
        pixels = image_array.reshape(-1, 4)
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)
        
        # Get cluster centers
        centers = np.uint8(kmeans.cluster_centers_)
        
        # Find count of pixels for each centroid
        labels = kmeans.labels_
        counts = np.bincount(labels)
        
        # Get the index of the dominant color
        dominant_color_index = np.argmax(counts)
        
        # Get the color with maximum count
        dominant_color = centers[dominant_color_index]
        
        # Calculate the percentage of the dominant color
        total_pixels = np.sum(counts)
        dominant_color_percentage = 100 * counts[dominant_color_index] / total_pixels
        
        # Create a mask for the dominant color pixels
        mask = labels.reshape(image_array.shape[:2]) == dominant_color_index
        
        # Convert mask to uint8
        mask = mask.astype(np.uint8) * 255
        
        # Create a transparent red color
        red_color = (255, 0, 0, 128)  # Transparent red
        
        # Create a transparent mask with red for dominant color pixels
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                if mask[y, x] == 255:
                    draw.point((x, y), fill=red_color)
        
        # Create a mask for the border pixels
        border_mask = np.zeros_like(mask)
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=np.uint8)
        border = cv2.filter2D(mask, -1, kernel)
        border_mask[border > 0] = 255
        
        # Draw a black border around the dominant color pixels
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                if border_mask[y, x] == 255 and mask[y, x] != 255:
                    draw.point((x, y), fill=(0, 0, 0, 255))
        
        # Composite the original image and the colored mask
        blended_image = Image.alpha_composite(image.convert('RGBA'), overlay)
        
        # Check if the image mode is RGBA (indicating transparency)
        if blended_image.mode == 'RGBA':
            # Save the image as PNG
            blended_image.save("marked_image.png")
            print("Image saved as 'marked_image.png'.")
        else:
            # Convert blended image to RGB mode before saving as JPEG
            blended_image = blended_image.convert('RGB')
            # Save the image as JPEG
            blended_image.save("marked_image.jpg")
            print("Image saved as 'marked_image.jpg'.")
        
        return dominant_color, dominant_color_percentage, blended_image
    except Exception as e:
        print("Error:", e)
        return None, None, None




def process_image(image_path):
    try:
        valid_extensions = ['.jpg', '.jpeg', '.jfif', '.png', '.webp']
        if os.path.splitext(image_path)[1].lower() not in valid_extensions:
            raise ValueError("Invalid image format. Supported formats are .jpg, .jpeg, .jfif, .png, .webp")
        
        dominant_colors = find_dominant_colors(image_path, k=64, resize_factor=0.5)

        if dominant_colors is not None:
            # Display the top dominant colors with their RGB values, hex codes, percentage, and color names
            max_percentage = 0
            max_percentage_color = None
            print("\nOther Colors Succeeding the DOMINANT COLOR are:")
            for color in dominant_colors:
                rgb_value = tuple(color[0])
                hex_value = webcolors.rgb_to_hex(rgb_value)
                percentage = color[1]
                name = color[2]
                if percentage > max_percentage:
                    max_percentage = percentage
                    max_percentage_color = (rgb_value, hex_value, name)
                else:
                    print(f"Color: {rgb_value}, Hex: {hex_value}, Percentage: {percentage:.2f}%, Name: {name}")
            
            # Display the highest dominant color as "DOMINANT COLOR" above other colors
            if max_percentage_color:
                print("\n              ---------------------------------------------------              ")
                print(f"DOMINANT COLOR: {max_percentage_color[0]}, Hex: {max_percentage_color[1]}, Percentage: {max_percentage:.2f}%, Name: {max_percentage_color[2]}")
                print("              ---------------------------------------------------              ")
                
            # Display the image with the highest dominant color marked as transparent red with a black border
            _, _, marked_image = mark_highest_dominant_color(image_path, k=64)
            
            # Save the image as PNG if transparency is involved
            if marked_image.mode == 'RGBA':
                marked_image.save("marked_image.png")
                print("Image saved as 'marked_image.png'.")
            else:
                # If the image doesn't contain transparency, save as JPEG
                marked_image.save("marked_image.jpg")
                print("Image saved as 'marked_image.jpg'.")
        else:
            print('Failed to find the dominant colors.')
    except ValueError as ve:
        print("ValueError:", ve)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    while True:
        image_path = input("\nEnter the URL or local path of the image to find dominant colors (or 'exit' to quit): ")
        if image_path.lower() == 'exit':
            break
        if image_path.startswith('http'):
            try:
                response = requests.get(image_path)
                if response.status_code == 200:
                    image_bytes = BytesIO(response.content)
                    image = Image.open(image_bytes)
                    # Save the image locally with a temporary name
                    temp_image_path = 'temp_image.png' if image.format == 'PNG' else 'temp_image.jpg'
                    image.save(temp_image_path)
                    # Process the image
                    process_image(temp_image_path)
                    # Delete the temporary image file
                    os.remove(temp_image_path)
                else:
                    print("Failed to fetch the image from the URL.")
            except Exception as e:
                print("Error:", e)
        else:
            if not os.path.exists(image_path):
                print("Invalid local path. Please provide a valid local path.")
            else:
                # Process the image
                process_image(image_path)