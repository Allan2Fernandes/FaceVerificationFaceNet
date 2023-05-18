import numpy as np
import sys

import torch
import SiameseNetworkProject

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

def get_encoding_from_path(path, device):
    encoding_file = open(path, 'r')
    lines = encoding_file.readlines()
    lines = [np.float32(line.strip()) for line in lines]
    encoding = torch.tensor(lines)
    encoding = torch.unsqueeze(encoding, dim=0)
    encoding = encoding.to(device)
    return encoding

#Load in the classifier
classifier = SiameseNetworkProject.SiameseNetworkProject(device)
classifier.load_state_dict(torch.load('../SavedModels/classifier.pt'))

#Build encoding1
encoding1_path = sys.argv[1]
encoding1 = get_encoding_from_path(encoding1_path, device)
#Build encoding2
encoding2_path = sys.argv[2]
encoding2 = get_encoding_from_path(encoding2_path, device)

classification_score = classifier(encoding1, encoding2)
print(classification_score)