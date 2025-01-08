from flask import Flask, render_template, request, redirect, url_for, make_response
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
import numpy as np
import os
import warnings
import requests
from io import BytesIO
import webcolors
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Maximum 16MB allowed

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

color_names_to_hex = webcolors.CSS3_NAMES_TO_HEX
color_rgbs = [webcolors.hex_to_rgb(hex_val) for hex_val in color_names_to_hex.values()]
color_names = list(color_names_to_hex.keys())

def rgb_to_name(rgb_triplet):
    min_distance = float('inf')
    closest_name = None
    
    for name, rgb in zip(color_names, color_rgbs):
        distance = np.linalg.norm(np.array(rgb_triplet) - np.array(rgb))
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    
    return closest_name

def find_dominant_colors(image_path, k=8, resize_factor=0.1, num_runs=3):
    try:
        image = Image.open(image_path)
        target_width, target_height = 300, 300
        image = image.resize((target_width, target_height))
        image = image.convert('RGB')
        image_array = np.array(image)
        pixels = image_array.reshape(-1, 3)
        
        best_dominant_colors = None
        
        for _ in range(num_runs):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
            kmeans.fit(pixels)
            centers = np.uint8(kmeans.cluster_centers_)
            labels = kmeans.labels_
            counts = np.bincount(labels)
                
            sorted_centers = centers[np.argsort(-counts)]
            sorted_counts = np.sort(counts)[::-1]
                
            total_pixels = np.sum(counts)
            percentages = sorted_counts / total_pixels * 100
                
            dominant_colors = list(zip(sorted_centers, percentages, [rgb_to_name(color) for color in sorted_centers]))
                
            dominant_color_index = np.argmax(counts)
            highest_dominant_color = centers[dominant_color_index]
            highest_dominant_color_percentage = 100 * counts[dominant_color_index] / total_pixels
            if not any(np.array_equal(color, highest_dominant_color) for color, _, _ in dominant_colors):
                color_name = rgb_to_name(highest_dominant_color)
                hex_val = webcolors.rgb_to_hex(highest_dominant_color)
                dominant_colors.append((highest_dominant_color, highest_dominant_color_percentage, color_name, hex_val))
                
            dominant_colors.sort(key=lambda x: x[1], reverse=True)
                
            unique_color_names = set()
            unique_dominant_colors = []
            for color in dominant_colors[:5]:
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
        image = Image.open(image_path)
        image = image.convert('RGB')
        image_array = np.array(image)
        pixels = image_array.reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)
        centers = np.uint8(kmeans.cluster_centers_)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        
        dominant_color_index = np.argmax(counts)
        dominant_color = centers[dominant_color_index]
        total_pixels = np.sum(counts)
        dominant_color_percentage = 100 * counts[dominant_color_index] / total_pixels
        
        mask = labels.reshape(image_array.shape[:2]) == dominant_color_index
        mask = mask.astype(np.uint8) * 255
        
        red_color = (255, 0, 0, 128)
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                if mask[y, x] == 255:
                    draw.point((x, y), fill=red_color)
        
        border_mask = np.zeros_like(mask)
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=np.uint8)
        border = cv2.filter2D(mask, -1, kernel)
        border_mask[border > 0] = 255
        
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                if border_mask[y, x] == 255 and mask[y, x] != 255:
                    draw.point((x, y), fill=(0, 0, 0, 255))
        
        blended_image = Image.alpha_composite(image.convert('RGBA'), overlay)
        
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
            max_percentage = 0
            max_percentage_color = None
            for color in dominant_colors:
                rgb_value = tuple(color[0])
                hex_value = webcolors.rgb_to_hex(rgb_value)
                percentage = color[1]
                name = color[2]
                if percentage > max_percentage:
                    max_percentage = percentage
                    max_percentage_color = (rgb_value, hex_value, name, percentage)
                else:
                    print(f"Color: {rgb_value}, Hex: {hex_value}, Percentage: {percentage:.2f}%, Name: {name}")

            if max_percentage_color:
                print("\n              ---------------------------------------------------              ")
                print(f"DOMINANT COLOR: {max_percentage_color[0]}, Hex: {max_percentage_color[1]}, Percentage: {max_percentage_color[3]:.2f}%, Name: {max_percentage_color[2]}")
                print("              ---------------------------------------------------              ")

            _, _, marked_image = mark_highest_dominant_color(image_path, k=64)
            
            if marked_image.mode == 'RGBA':
                marked_image_rgb = marked_image
                marked_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'marked_' + os.path.basename(image_path))
                marked_image_rgb.save(marked_image_path.replace('.png', '.png'), format='PNG')

            else:
                # Convert RGBA image to RGB
                marked_image_rgb = marked_image.convert('RGB')
            
                marked_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'marked_' + os.path.basename(image_path))
                marked_image_rgb.save(marked_image_path.replace('.jpg', '.jpg'), format='JPEG')

            return max_percentage_color[0], max_percentage_color[3], marked_image_path, dominant_colors
        else:
            print('Failed to find the dominant colors.')
    except ValueError as ve:
        print("ValueError:", ve)
    except Exception as e:
        print("Error:", e)
    return None, None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' in request.files:
            # File upload from PC
            file = request.files['file']

            if file.filename == '':
                return redirect(request.url)

            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                dominant_color, dominant_color_name, marked_image_path, dominant_colors = process_image(filepath)

                if dominant_color:
                    marked_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'marked_' + filename)

                    # Pass hex value along with other dominant color attributes
                    dominant_colors_with_hex = [(color[0], color[1], color[2], webcolors.rgb_to_hex(color[0])) for color in dominant_colors]

                    os.remove(filepath)
                    return render_template('result.html', dominant_color=dominant_color, dominant_color_name=dominant_color_name, marked_image_path=marked_image_path, dominant_colors=dominant_colors_with_hex)
                else:
                    os.remove(filepath)
                    return "Failed to process the image."
            else:
                return "Invalid file format."
        elif 'url' in request.form:
            # Image URL provided
            image_url = request.form['url']
            response = requests.get(image_url)
            if response.status_code == 200:
                image_bytes = BytesIO(response.content)
                image = Image.open(image_bytes)
                image_format= image.format
                temp_image_path = 'temp_image.jpg' if image.format in ['JPEG', 'JPG', 'JFIF', 'WEBP'] else 'temp_image.png' # Save the image temporarily
                image.save(temp_image_path)
                dominant_color, dominant_color_name, marked_image_path, dominant_colors = process_image(temp_image_path)
                os.remove(temp_image_path)  # Delete the temporary image file

                if dominant_color:
                    # Pass hex value along with other dominant color attributes
                    dominant_colors_with_hex = [(color[0], color[1], color[2], webcolors.rgb_to_hex(color[0])) for color in dominant_colors]

                    return render_template('result.html', dominant_color=dominant_color, dominant_color_name=dominant_color_name, marked_image_path=marked_image_path, dominant_colors=dominant_colors_with_hex)
                else:
                    return "Failed to process the image."
            else:
                return "Failed to fetch the image from the URL."
        else:
            return "Invalid request."
    except Exception as e:
        print("Error:", e)
        return render_template('result.html', dominant_color=dominant_color, dominant_color_name=dominant_color_name, marked_image_path=marked_image_path, dominant_colors=dominant_colors)



if __name__ == '__main__':
    app.run(debug=True)
