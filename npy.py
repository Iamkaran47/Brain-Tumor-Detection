# import numpy as np
# import pickle
# import os
#
# # Replace 'your_npy_file.npy' with the actual path to your npy file
# npy_file_path = 'images.npy'
#
# # Load the npy file
#
# data = np.load(npy_file_path, allow_pickle=True)
# # View the contents of the npy file
# print(data)
# # View the first few elements (e.g., first 10 elements) of the array
# # print(data[:10])
# #
# # # View specific elements (e.g., elements from index 100 to 110)
# # print(data[100:111])
#
import numpy as np
from PIL import Image

# Replace 'your_npy_file.npy' with the actual path to your npy file
npy_file_path = 'labels.npy'

# Load the npy file with allow_pickle=True
data = np.load(npy_file_path, allow_pickle=True)

# Assuming data is a list of NumPy arrays representing images
# Iterate through the loaded data and save each image
for i, img_array in enumerate(data):
    # Convert the NumPy array back to an image
    img = Image.fromarray((img_array * 255).astype(np.uint8))  # Assuming pixel values were normalized to [0, 1]

    # Replace 'output_directory' with the directory where you want to save the images
    img_path = f'images/image_{i}.png'  # You can use any format like .jpg, .png, etc.

    # Save the image
    img.save(img_path)

print("Images extracted and saved successfully.")
