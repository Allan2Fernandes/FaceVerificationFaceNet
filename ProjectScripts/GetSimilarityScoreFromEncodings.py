"""
To execute this file, run the following command:
python GetSimilarityScoreFromEncoding.py encoding1.txt encoding2.txt

The script will feed both encodings to a classifier model which will return a score from 0-1
0 means the 2 encodings are of 2 different people
1 means the encodings are of the same person.
"""

import numpy as np
import sys
import torch
import SiameseNetworkProject
import os

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
current_file_path = os.path.realpath(__file__)
parent_path = os.path.dirname(current_file_path)
classifier = SiameseNetworkProject.SiameseNetworkProject(device)
classifier.load_state_dict(torch.load(os.path.join(parent_path, '../SavedModelsV1/m_hidden_layers/Iteration10000Weights.pt'), map_location=device))

#Build encoding1
encoding1_path = sys.argv[1]
encoding1 = get_encoding_from_path(encoding1_path, device)
#Build encoding2
encoding2_path = sys.argv[2]
encoding2 = get_encoding_from_path(encoding2_path, device)
classifier.eval()
classifier.to(device=device)
classification_score = classifier(encoding1, encoding2)
print(classification_score.item())