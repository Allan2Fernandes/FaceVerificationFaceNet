import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from torchvision import transforms

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

#Initialize the encoding and detection models and constants
encoding_detection_model = MTCNN(
    image_size=128, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device='cpu', select_largest=True
)
bounding_box_detection_model = MTCNN(keep_all=True, device='cpu', select_largest=False)


# For a model pretrained on VGGFace2
encoding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device=device)

# Load the classifier network
classifier_network = torch.load("SavedModels/Iteration10000")

threshold = 0.7
face_is_encoded = False


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
    transforms.CenterCrop(128)
])

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
    """
    LOOK DIRECTLY INTO THE CAMERA AT THE START SO IT GETS AN ACCURATE BASELINE ENCODING
    Use the first frame to get an encoding of the person's face. 
    Then save that encoding. This encoding is what future images will be compared to. 
    Normally this is the encoding that would be saved in the database and then compared to later
    """
    try:
        rval, frame = vc.read()
        # Get the cropped image
        cropped_image = encoding_detection_model(frame)
        # Get the encoding of the cropped image
        encoding_saved_image = encoding_model(torch.unsqueeze(cropped_image.to(device=device), dim=0))
        face_is_encoded = True
    except:
        print("No baseline encoding available. Make sure it has a face to encode at the very start of the program so it has an encoding to compare to.")
else:
    rval = False

while rval:
    rval, frame = vc.read()
    boxed_frame = frame
    try:
        #Get the bounding box
        print(frame)
        boxed_frame = get_frame_with_bounding_boxes(frame)
        #Get the cropped image
        cropped_image = encoding_detection_model(frame)
        #Get the encoding of the cropped image
        image_encoding = encoding_model(torch.unsqueeze(cropped_image.to(device=device), dim=0))
        #Get the encoding distances
        # encoding_distance = get_encoding_distance(image_encoding, encoding_saved_image)
        prediction = classifier_network(encoding_saved_image, image_encoding)
        #Determine if the person is verified
        print("Person is verified: {0}".format(torch.squeeze(prediction)))
    except:
        if not face_is_encoded:
            print("A face was not encoded at the start of the program")
        else:
            print("Couldn't encode face")
        pass

    cv2.imshow("Face verification demo", boxed_frame)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

vc.release()
cv2.destroyWindow("Face verification demo")
