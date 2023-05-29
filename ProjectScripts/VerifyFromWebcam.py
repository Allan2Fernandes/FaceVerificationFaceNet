"""
To execute the script, run the command
python VerifyFromWebcam.py encoding_example.txt

where encoding_example.txt is a text file with the encoding values for the person's face you want to verify
"""


import sys
import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from torchvision import transforms
import numpy as np
import SiameseNetworkProject

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

device = torch.device('cpu')
cropped_image_size = 160
#Initialize the encoding and detection models and constants
encoding_detection_model = MTCNN(
    image_size=cropped_image_size, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device='cpu', select_largest=True
)
bounding_box_detection_model = MTCNN(keep_all=True, device='cpu', select_largest=False)


# For a model pretrained on VGGFace2
encoding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device=device)

#Load in the classifier
classifier = SiameseNetworkProject.SiameseNetworkProject(device)
classifier.load_state_dict(torch.load('../SavedModelsV1/m_hidden_layers/Iteration10000Weights.pt', map_location=device))
classifier.eval()
classifier.to(device=device)
threshold = 0.7
#Feel free to modify this value
verification_threshold = 0.95


def get_encoding_distance(encoding_unverified, encoding_saved):
    return (encoding_saved - encoding_unverified).norm()

def determine_if_verified(encoding_distance, threshold):
    print("Distance = {0}".format(encoding_distance))
    return encoding_distance <= threshold

def get_frame_with_bounding_boxes(frame):
    prediction = bounding_box_detection_model.detect(frame)
    boxes, _ = prediction
    #Draw all the boxes
    for box in boxes:
        #Draw each box
        new_frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)
    return new_frame

convert_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(cropped_image_size)
])

def get_encoding_from_path(path, device):
    encoding_file = open(path, 'r')
    lines = encoding_file.readlines()
    lines = [np.float32(line.strip()) for line in lines]
    encoding = torch.tensor(lines)
    encoding = torch.unsqueeze(encoding, dim=0)
    encoding = encoding.to(device)
    return encoding

encoding_saved_image = get_encoding_from_path(sys.argv[1], device)

cv2.namedWindow("Face verification demo")
vc = cv2.VideoCapture(0)
"""
Check max dimensions before setting width and height. Use the 2 lines below to determine max dimensions
width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
"""
width = min(480, vc.get(cv2.CAP_PROP_FRAME_WIDTH))
height = min(480, vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
vc.set(3, width)
vc.set(4, height)

if vc.isOpened():
    try:
        rval, frame = vc.read()
    except:
        print("Webcam error")
else:
    rval = False

while rval:
    rval, frame = vc.read()
    boxed_frame = frame

    try:
        # Get the bounding box
        boxed_frame = get_frame_with_bounding_boxes(frame)
        # Get the cropped image
        cropped_image = encoding_detection_model(frame)
        # Get the encoding of the cropped image
        image_encoding = encoding_model(torch.unsqueeze(cropped_image.to(device=device), dim=0))
        # Get the encoding distances
        # encoding_distance = get_encoding_distance(image_encoding, encoding_saved_image)
        prediction = classifier(encoding_saved_image, image_encoding)
        # Determine if the person is verified
        prediction_value = torch.squeeze(prediction)
        prediction_value = prediction_value.item()
        print("Verification prediction: {0}".format(
            prediction_value))  # Recommend verifying when the prediction is over 90-95%
    except:
        print("Couldn't find face")
    cv2.imshow("Face verification demo", boxed_frame)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

vc.release()
cv2.destroyWindow("Face verification demo")
