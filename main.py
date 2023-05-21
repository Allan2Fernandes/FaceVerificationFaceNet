"""
Model -     Siamese network with each leg being Facenet.
            Create an encoding for each leg and then get the difference.
            The difference should be of the dimension [None, 512].
            Connect this layer to a dense layer with 1 neuron directly or through a hidden layer
            The labels are 0s and 1s for if the images encoded were of the same person or not
            Sigmoid activation and a binary crossentropy loss with adam optimizer

Dataset -   it will be a tuple of batch x (img1, img2, label)
            If the 2 images are of the same person, label = 1. If not, label = 0
            To create the dataset, randomly select an image of a certain person
            and then randomly select an image of another person or of the same person.
"""
import sys
import torch
import SiameseNetworkV1 as sn
import DatasetBuilder as db
import TrainModel as tm

base_directory_path = "D:/Datasets/vggface2_224"
batch_size = 32
device = torch.device('cuda')

siamese_network = sn.SiameseNetworkV1(device=device)
database_builder = db.DatasetBuilder(base_directory_path=base_directory_path, batch_size=batch_size, device=device)
model_trainer = tm.TrainModel(dataset_builder=database_builder, siamese_network=siamese_network, device=device)
model_trainer.initialize_optimizer_loss_functions()
model_trainer.train_model(iterations=10000, directory_to_save="SavedModelsV1/m_hidden_layers", device=device)