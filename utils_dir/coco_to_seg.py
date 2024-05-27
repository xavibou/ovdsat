import os
import json
import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

def coco_to_seg(annotation_file, image_directory, save_path):
    '''
    Convert COCO annotations to segmentation masks in class directories for prototype initialization.

    Args:
        annotation_file (str): Path to the COCO annotation file.
        image_directory (str): Path to the directory containing the images.
        save_path (str): Path to the directory where the segmentation masks will be saved.
    '''
    # Create the output directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load the COCO annotation file
    coco = COCO(annotation_file)

    # Get the list of category IDs
    category_ids = coco.getCatIds()


    # Loop through each category
    for category_id in category_ids:
        # Get the category information
        category_info = coco.loadCats(category_id)[0]
        category_name = category_info['name']
        category_directory = os.path.join(save_path, category_name)

        # Create a directory for the category
        if not os.path.exists(category_directory):
            os.makedirs(category_directory)

        # Get the image IDs containing the selected category
        image_ids = coco.getImgIds(catIds=category_id)

        for image_id in image_ids:
            # Load the image and annotations
            image_info = coco.loadImgs(image_id)[0]
            image_filename = image_info['file_name']
            image = cv2.imread(os.path.join(image_directory, image_filename))
            annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id, catIds=category_id))

            # Create a mask for the selected category
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for annotation in annotations:
                bbox = list(map(int, annotation['bbox']))
                x, y, w, h = bbox
                mask[y:y + h, x:x + w] = 255

            # Save the image and mask
            image_filename_without_ext = os.path.splitext(image_filename)[0]
            mask_filename = f"{image_filename_without_ext}.mask{os.path.splitext(image_filename)[1]}"
            cv2.imwrite(os.path.join(category_directory, image_filename), image)
            cv2.imwrite(os.path.join(category_directory, mask_filename), mask)

        print(f"Processed category: {category_name}")

    print("Processing complete.")