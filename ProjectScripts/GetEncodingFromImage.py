"""
To use this script, run the command
python GetEncodingFromImage.py img1.png encoding1.txt

The script will open the png file, detect a face and encode the face in img1.png
and then write the encoding to encoding1.txt
"""

import sys
import torch
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
#sys.argv[1] = image file path

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

#classifier_model_path = "../SavedModels/Iteration10000"

#Initialize the encoding and detection models and constants
encoding_detection_model = MTCNN(
    image_size=128, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device='cpu', select_largest=True
)

# For a model pretrained on VGGFace2
encoding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device=device)

# Load the classifier network
#classifier_network = torch.load(classifier_model_path)

def convert_image_file_to_tensor(image_file):
    convert_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(128),
        transforms.CenterCrop(128) #Center crop the image
    ])
    return convert_tensor(image_file)

def detect_face_in_image(image_tensor):
    cropped_image = encoding_detection_model(image_tensor, return_prob = False)
    return cropped_image

def encode_face(cropped_image):
    encoding = encoding_model(torch.unsqueeze(cropped_image.to(device=device), dim=0))
    return encoding

image_path = sys.argv[1]

image_file = Image.open(image_path)
#Convert to tensor and then center crop the image
image_tensor = convert_image_file_to_tensor(image_file=image_file)
permuted_image = (torch.permute(image_tensor, (1, 2, 0)) * 255).int()
permuted_image = permuted_image[:,:,:3]
# Crop the face
cropped_image = detect_face_in_image(image_tensor=permuted_image)
#Encode the image
encoded_image = encode_face(cropped_image=cropped_image)
#Write to a text file
file = open(sys.argv[2], 'w')
for i in range(512):
    file.write(str(encoded_image[:, i].item()))
    file.write('\n')
file.close()




