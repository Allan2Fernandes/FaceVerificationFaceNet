import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False
import os
import random
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1

class DatasetBuilder:
    def __init__(self, base_directory_path, batch_size, device):
        self.base_directory_path = base_directory_path
        self.list_classes = os.listdir(base_directory_path)
        self.batch_size = batch_size
        self.device = device
        self.transform_image = transforms.Compose([
            transforms.Resize(160),
            transforms.ToTensor()
        ])
        pass

    def build_positive_batch(self):
        left_images = []
        right_images = []
        for _ in range(self.batch_size):
            # Pick a random class
            class_name = self.get_random_class()
            # Get class path
            class_path = self.get_class_path(class_name=class_name)
            # Get all possible list of images
            list_images = os.listdir(class_path)
            # Pick a random image in the class for left network
            left_image = self.get_random_image(list_of_images=list_images)
            # Pick a second random image excluding the first in the class for the right network
            right_image = self.get_random_image_excluding_one(list_of_images=list_images, excluded_element=left_image)
            # Label = 1
            # Convert to tensors
            left_image = self.transform_image(Image.open(os.path.join(class_path, left_image))).to(self.device)
            right_image = self.transform_image(Image.open(os.path.join(class_path, right_image))).to(self.device)
            left_images.append(torch.unsqueeze(left_image, dim=0))
            right_images.append(torch.unsqueeze(right_image, dim=0))
            pass
        left_dataset = torch.cat(left_images, dim = 0)
        right_dataset = torch.cat(right_images, dim = 0)
        label_dataset = torch.ones(size=(self.batch_size, 1))
        return left_dataset, right_dataset, label_dataset

    def build_negative_batch(self):
        left_images = []
        right_images = []
        for _ in range(self.batch_size):
            # Pick a random class
            class1_name = self.get_random_class()
            # Get class path
            class1_path = self.get_class_path(class_name=class1_name)
            # Get all possible list of images
            list_images_class1 = os.listdir(class1_path)
            # Pick a random image in that class for the left network
            left_image = self.get_random_image(list_of_images=list_images_class1)
            # Convert to tensors
            left_image = self.transform_image(Image.open(os.path.join(class1_path, left_image))).to(self.device)

            # Pick a second random class excluding the previous class
            class2_name = self.get_random_class_excluding_one(excluded_element=class1_name)
            # Get class path
            class2_path = self.get_class_path(class_name=class2_name)
            # Get all possible list of images
            list_images_class2 = os.listdir(class2_path)
            # Pick a random image in this class for the right network
            right_image = self.get_random_image(list_of_images=list_images_class2)
            # Convert to tensors
            right_image = self.transform_image(Image.open(os.path.join(class2_path, right_image))).to(self.device)

            left_images.append(torch.unsqueeze(left_image, dim=0))
            right_images.append(torch.unsqueeze(right_image, dim=0))
            pass
        left_dataset = torch.cat(left_images, dim=0)
        right_dataset = torch.cat(right_images, dim=0)
        label_dataset = torch.zeros(size=(self.batch_size, 1))
        return left_dataset, right_dataset, label_dataset

    def get_batch(self):
        # Get a positive batch
        left_positive_batch, right_positive_batch, positive_labels = self.build_positive_batch()
        # Get a negative batch
        left_negative_batch, right_negative_batch, negative_labels = self.build_negative_batch()
        # Concatenate batches and shuffle?
        left_batch = torch.cat([left_positive_batch, left_negative_batch], dim=0)
        right_batch = torch.cat([right_positive_batch, right_negative_batch], dim=0)
        labels = torch.cat([positive_labels, negative_labels], dim=0).to(self.device)
        # Return batch
        return left_batch, right_batch, labels

    def get_class_path(self, class_name):
        return os.path.join(self.base_directory_path, class_name)

    def get_random_class(self):
        return random.choice(self.list_classes)

    def get_random_class_excluding_one(self, excluded_element):
        return random.choice([element for element in self.list_classes if element != excluded_element])

    def get_random_image(self, list_of_images):
        return random.choice(list_of_images)

    def get_random_image_excluding_one(self, list_of_images, excluded_element):
        return random.choice([element for element in list_of_images if element != excluded_element])

    pass
