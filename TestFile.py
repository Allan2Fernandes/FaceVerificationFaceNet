from PIL import Image
import numpy as np
image_path = "D:/Datasets/vggface2_224/n000007/0011_01.jpg"

image_file = Image.open(image_path)
image_array = np.asarray(image_file)
print(image_array.shape)