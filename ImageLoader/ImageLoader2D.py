# import glob
# import numpy as np
# from PIL import Image
# from skimage.io import imread
# from tqdm import tqdm

# # Set the path to your dataset directory
# folder_path = "../Datasets/Kvasir-SEG/"

# def load_data(img_height, img_width, images_to_be_loaded, dataset, split='train'):
#     IMAGES_PATH = f"{folder_path}{split}/images/"
#     MASKS_PATH = f"{folder_path}{split}/masks/"

#     # Initialize train_ids
#     train_ids = []

#       # Print the dataset string being passed to the function
#     print(f"Dataset string passed: '{dataset}'")  # Debug statement
#     print(f"Expected: 'Datasets/Kvasir-SEG'")

#     # Sanitize the dataset input
#     dataset_sanitized = dataset.strip().lower()
#     print(f"Sanitized dataset: '{dataset_sanitized}'")  # Debug statement

#     if dataset_sanitized == 'datasets/kvasir-seg'.lower():
#         print("Dataset condition matched for 'Datasets/Kvasir-SEG'")
#         train_ids = glob.glob(IMAGES_PATH + "*.jpg") + glob.glob(IMAGES_PATH + "*.png")  # Load both jpg and png images
#     elif dataset.lower() == 'cvc-clinicdb':
#         train_ids = glob.glob(IMAGES_PATH + "*.tif")
#     elif dataset.lower() in ['cvc-colondb', 'etis-laribpolypdb']:
#         train_ids = glob.glob(IMAGES_PATH + "*.png")
#     else:
#         raise ValueError("Unsupported dataset. Please choose 'Kvasir-SEG', 'cvc-clinicdb', 'cvc-colondb', or 'etis-laribpolypdb'.")

#     # Check if any images were found
#     if not train_ids:
#         raise FileNotFoundError(f"No images found in {IMAGES_PATH}. Please check the directory.")

#     # Determine how many images to load
#     if images_to_be_loaded == -1:
#         images_to_be_loaded = len(train_ids)

#     X_train = np.zeros((images_to_be_loaded, img_height, img_width, 3), dtype=np.float32)
#     Y_train = np.zeros((images_to_be_loaded, img_height, img_width), dtype=np.uint8)

#     print('Resizing training images and masks: ' + str(images_to_be_loaded))
#     for n, id_ in tqdm(enumerate(train_ids)):
#         if n == images_to_be_loaded:
#             break

#         image_path = id_
#         mask_path = image_path.replace("images", "masks")  # Assuming masks are in the 'masks' folder

#         # Load the images
#         image = imread(image_path)
#         mask_ = imread(mask_path)

#         # Resize and normalize the image
#         pillow_image = Image.fromarray(image)
#         pillow_image = pillow_image.resize((img_height, img_width))
#         image = np.array(pillow_image) / 255.0  # Normalize image to [0, 1]

#         X_train[n] = image

#         # Resize and process the mask
#         pillow_mask = Image.fromarray(mask_)
#         pillow_mask = pillow_mask.resize((img_height, img_width), resample=Image.LANCZOS)
#         mask_ = np.array(pillow_mask)

#         # Create binary mask using vectorized NumPy operation
#         mask = (mask_ >= 127).astype(np.bool_)

#         Y_train[n] = mask

#     Y_train = np.expand_dims(Y_train, axis=-1)  # Expand dims for compatibility

#     return X_train, Y_train


import glob
import numpy as np
from PIL import Image
from skimage.io import imread
from tqdm import tqdm

# Set the path to your dataset directory
folder_path = "Datasets/Kvasir-SEG/"  # Make sure this path is correct relative to your working directory

def load_data(img_height, img_width, images_to_be_loaded, dataset, split='train'):
    IMAGES_PATH = f"{folder_path}{split}/images/"
    MASKS_PATH = f"{folder_path}{split}/masks/"

    # Initialize train_ids
    train_ids = []

    # Debugging prints
    # print(f"Dataset string passed: '{dataset}'")
    # print(f"Expected: 'Datasets/Kvasir-SEG'")

    # Sanitize and lower the dataset input for comparison
    dataset_sanitized = dataset.strip().lower()
    # print(f"Sanitized dataset: '{dataset_sanitized}'")  # Debug statement

    # Match dataset and populate train_ids
    if 'kvasir-seg' in dataset_sanitized:
        # print("Dataset condition matched for 'Datasets/Kvasir-SEG'")
        train_ids = glob.glob(IMAGES_PATH + "*.jpg") + glob.glob(IMAGES_PATH + "*.png")  # Load both jpg and png images
    elif 'cvc-clinicdb' in dataset_sanitized:
        train_ids = glob.glob(IMAGES_PATH + "*.tif")
    elif dataset_sanitized in ['cvc-colondb', 'etis-laribpolypdb']:
        train_ids = glob.glob(IMAGES_PATH + "*.png")
    else:
        raise ValueError("Unsupported dataset. Please choose 'Kvasir-SEG', 'cvc-clinicdb', 'cvc-colondb', or 'etis-laribpolypdb'.")


    print(f"Total number of images found: {len(train_ids)}")

    # Ensure that train_ids is populated
    if not train_ids:
        raise FileNotFoundError(f"No images found in {IMAGES_PATH}. Please check the directory.")

    # Determine how many images to load
    if images_to_be_loaded == -1:
        images_to_be_loaded = len(train_ids)

    X_train = np.zeros((images_to_be_loaded, img_height, img_width, 3), dtype=np.float32)
    Y_train = np.zeros((images_to_be_loaded, img_height, img_width), dtype=np.uint8)

    print(f'Resizing training images and masks: {images_to_be_loaded}')
    for n, id_ in tqdm(enumerate(train_ids)):
        if n == images_to_be_loaded:
            break

        image_path = id_
        mask_path = image_path.replace("images", "masks")  # Assuming masks are in the 'masks' folder

        # Load the images
        image = imread(image_path)
        mask_ = imread(mask_path)


        # Resize and normalize the image
        pillow_image = Image.fromarray(image)
        pillow_image = pillow_image.resize((img_height, img_width))
        image = np.array(pillow_image) / 255.0  # Normalize image to [0, 1]

        X_train[n] = image

        # Resize and process the mask
        pillow_mask = Image.fromarray(mask_)
        pillow_mask = pillow_mask.resize((img_height, img_width), resample=Image.LANCZOS)
        mask_ = np.array(pillow_mask)

        # Create binary mask using vectorized NumPy operation
        mask = (mask_ >= 127).astype(np.bool_)

        Y_train[n] = mask

    Y_train = np.expand_dims(Y_train, axis=-1)  # Expand dims for compatibility

    return X_train, Y_train

