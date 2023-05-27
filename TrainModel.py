import torch
from facenet_pytorch import InceptionResnetV1
from torcheval.metrics import BinaryAccuracy

class TrainModel:
    def __init__(self, dataset_builder, siamese_network, device):
        self.dataset_builder = dataset_builder
        self.siamese_network = siamese_network
        self.device = device
        self.siamese_network.to(device)
        self.learning_rate0 = 0.0001
        self.learning_rate1 = 0.00005
        self.learning_rate2 = 0.00002
        self.learning_rate3 = 0.000008
        self.selected_learning_rate = self.learning_rate0
        print("Number of parameters in the model: {0}".format(self.get_n_params(siamese_network)))
        self.losses = []
        pass

    def get_n_params(self, model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def initialize_optimizer_loss_functions(self):
        self.loss_function = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.siamese_network.parameters(), lr=self.selected_learning_rate)      # 0.0001
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
        pass

    def set_new_learning_rate(self, new_learning_rate):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_learning_rate

    def train_model(self, iterations, directory_to_save, device):
        encoder_model = InceptionResnetV1(pretrained='vggface2').eval().to(device=device)
        self.freeze_model(encoder_model)
        for iteration in range(1, iterations+1):
            # Save the model every 5000 iterations
            if (iteration) % 2000 == 0 and iteration > 0:
                torch.save(self.siamese_network.state_dict(), f"{directory_to_save}/Iteration{iteration}Weights.pt")
                pass

            # Reset the metric every few iterations
            if iteration % 100 == 0:
                self.reset_metric()
                self.losses = []
                pass

            # Reset the optimizer gradients
            self.optimizer.zero_grad()

            # Get the batch
            try:
                left_batch, right_batch, labels = self.dataset_builder.get_batch()
            except:
                print("Truncated image error")
                continue

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
            self.losses.append(loss)
            accuracy = self.binary_accuracy_metric(siamese_network_prediction, labels)
            average_loss = torch.mean(torch.stack(self.losses)).item()
            print("Loss for iteration: {0} = {1} || Accuracy = {2} || lr = {3}".format(iteration, average_loss, accuracy, self.selected_learning_rate))
            # Differentiate loss with respect to parameters
            loss.backward()
            # Gradient descent step
            self.optimizer.step()

            if iteration%100 > 60:
                if average_loss < 0.16 and self.selected_learning_rate == self.learning_rate0:
                    self.selected_learning_rate = self.learning_rate1
                    self.set_new_learning_rate(self.selected_learning_rate)
                elif average_loss < 0.14 and self.selected_learning_rate == self.learning_rate1:
                    self.selected_learning_rate = self.learning_rate2
                    self.set_new_learning_rate(self.selected_learning_rate)
                elif average_loss < 0.12 and self.selected_learning_rate == self.learning_rate2:
                    self.selected_learning_rate = self.learning_rate3
                    self.set_new_learning_rate(self.selected_learning_rate)

            pass
        pass

    # noinspection DuplicatedCode
    def train_modelV3(self, iterations, directory_to_save, device):
        for iteration in range(iterations):
            # Save the model every 5000 iterations
            if iteration % 5000 == 0 and iteration > 0:
                torch.save(self.siamese_network.state_dict(), f"{directory_to_save}/Iteration{iteration}Weights.pt")
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

            # Feed the left and right batches into the classifier model
            siamese_network_prediction = self.siamese_network(left_batch, right_batch)

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
