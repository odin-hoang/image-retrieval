import os
import requests
import json
import numpy as np
from feature_extractor import extract_resnet101_features
def download_and_store_first_image(json_data, output_directory):
    image_urls = []

    for item in json_data:
        if 'images' in item and len(item['images']) > 0:
            first_image_url = item['images'][0]
            image_urls.append(first_image_url)

            # Download the image
            response = requests.get(first_image_url)
            if response.status_code == 200:
                # Create the output directory if it doesn't exist
                os.makedirs(output_directory, exist_ok=True)

                # Save the image to the specified directory
                image_filename = f"{item['name'].replace(' ', '_')}_image.jpg"
                image_path = os.path.join(output_directory, image_filename)
                with open(image_path, 'wb') as file:
                    file.write(response.content)

    return image_urls

# Load the JSON data from your file
with open('/content/dataChuot.json', 'r') as file:
    json_data = json.load(file)

# Specify the output directory for saving images
output_directory = '/content/images'
# Get the first image URL for each element and download the image to the specified directory
result_image_urls = download_and_store_first_image(json_data, output_directory)
print(result_image_urls)
image_directory = '/content/images'

# List to store extracted features
all_features = []

# Loop through each image in the directory and extract features
for filename in os.listdir(image_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_directory, filename)
        image_features = extract_resnet101_features(image_path)
        all_features.append(image_features)

# Convert the list of features to a NumPy array
all_features_array = np.array(all_features)

# Print or use the 'all_features_array' as needed
np.save('/content/drive/MyDrive/CS336/all_features_array.npy', all_features_array)