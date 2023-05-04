import os
import numpy as np
import cv2 as cv


def get_image_descriptor(image_path, hog):
    image = cv.imread(image_path, 0)
    if image is None or image.shape == (0, 0):
        raise ValueError(f'Invalid image: {image_path}')
    image_gaussian = cv.GaussianBlur(image, (3, 3), 0)
    image_resized = cv.resize(image_gaussian, (128, 128), interpolation=cv.INTER_AREA)
    return hog.compute(image_resized).T


def get_image_descriptors(base_dir, classes, hog):
    descriptors = np.empty((0, hog.getDescriptorSize()), dtype=np.float32)
    labels = []
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(base_dir, 'database', class_name)
        print(f'Processing class: {i} {class_name}')
        for image_name in os.listdir(class_dir):
            try:
                descriptor = get_image_descriptor(os.path.join(class_dir, image_name), hog)
                descriptors = np.vstack([descriptors, descriptor])
                labels.append(i)
            except ValueError as e:
                print(e)
    return descriptors, labels


def save_data(base_dir, data, labels):
    output_dir = os.path.join(base_dir, 'vector_data')
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'vector_training'), data)
    np.save(os.path.join(output_dir, 'vector_y'), labels)


if __name__ == '__main__':
    base_dir = os.getcwd()
    classes = os.listdir(os.path.join(base_dir, 'database'))
    hog = cv.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)
    descriptors, labels = get_image_descriptors(base_dir, classes, hog)
    save_data(base_dir, descriptors, labels)

    print('Data saved successfully!')
