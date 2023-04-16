import torch
import DatasetBuilder
import SiameseNetwork
from facenet_pytorch import InceptionResnetV1
from torcheval.metrics import BinaryAccuracy


class TrainModel:
    def __init__(self, dataset_builder: DatasetBuilder, siamese_network: SiameseNetwork, device: torch.device):
        self.dataset_builder = dataset_builder
        self.siamese_network = siamese_network
        self.device = device
        pass

    def initialize_optimizer_loss_functions(self):
        self.loss_function = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.siamese_network.parameters(), lr=0.0001)      # 0.0001
        self.metric = BinaryAccuracy(threshold=0.5, device=self.device)     # 0.5
        pass

    def freeze_model(self, encoder):
        for param in encoder.parameters():
            param.requires_grad = False
            pass
        pass

    def binary_accuracy_metric(self, y_true, y_pred):
        self.metric.update(torch.squeeze(y_true.to(self.device)), torch.squeeze(y_pred.to(self.device)))
        accuracy_score = self.metric.compute()
        return accuracy_score

    def reset_metric(self):
        self.metric.reset()

    def train_model(self, iterations, device):
        encoder_model = InceptionResnetV1(pretrained='vggface2').eval().to(device=device)
        self.freeze_model(encoder_model)
        for iteration in range(iterations):
            # Save the model every 5000 iterations
            if iteration % 1000 == 0 and iteration > 0:
                torch.save(self.siamese_network, f"SavedModels/Iteration{iteration}")
                pass

            # Reset the metric every few iterations
            if iteration % 100 == 0:
                self.reset_metric()

            # Reset the optimizer gradients
            self.optimizer.zero_grad()

            # Get the batch
            try:
                left_batch, right_batch, labels = self.dataset_builder.get_batch()
            except:
                print("Truncated image error")
                continue
            """
            # Block of code for testing the dataset
            index = 2
            left_image_test = left_batch[index]
            right_image_test = right_batch[index]
            label_test = labels[index]
            plt.imshow(torch.permute(left_image_test, (1, 2, 0)))
            plt.show()
            plt.imshow(torch.permute(right_image_test, (1, 2, 0)))
            plt.show()
            print(label_test)
            """
            # Encode the left and right encodings and don't use grad
            with torch.no_grad():
                left_encoding = encoder_model(left_batch)
                right_encoding = encoder_model(right_batch)
            # Use the siamese network to get a prediction
            siamese_network_prediction = self.siamese_network(left_encoding, right_encoding)
            """
            if iteration%50 == 0:
                print(siamese_network_prediction)
            """
            # Calculate loss on prediction
            loss = self.loss_function(siamese_network_prediction, labels)
            accuracy = self.binary_accuracy_metric(siamese_network_prediction, labels)
            print("Loss for iteration: {0} = {1} || Accuracy = {2}".format(iteration, loss, accuracy))
            # Differentiate loss with respect to parameters
            loss.backward()
            # Gradient descent step
            self.optimizer.step()
            pass
        pass
