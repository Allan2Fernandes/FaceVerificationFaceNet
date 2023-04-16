import requests
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

classifier_model_path = "../SavedModels/Iteration10000"

#Initialize the encoding and detection models and constants
encoding_detection_model = MTCNN(
    image_size=128, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device='cpu', select_largest=True
)

# For a model pretrained on VGGFace2
encoding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device=device)

# Load the classifier network
classifier_network = torch.load(classifier_model_path)


def get_img_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def plot_image(image):
    plt.imshow(image)
    plt.show()

def convert_image_file_to_tensor(image_file):
    convert_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(128),
        transforms.CenterCrop(128)
    ])
    return convert_tensor(image_file)

def detect_face_in_image(image_tensor):
    cropped_image = encoding_detection_model(image_tensor, return_prob = False)
    return cropped_image

def encode_face(cropped_image):
    encoding = encoding_model(torch.unsqueeze(cropped_image.to(device=device), dim=0))
    return encoding

def get_verification_determination(encoding1, encoding2):
    distance = (encoding2-encoding1).norm()
    classifier_prediction = classifier_network(encoding1, encoding2)
    return distance, classifier_prediction
