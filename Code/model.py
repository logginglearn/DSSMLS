import torch
import torch.nn as nn
import torch.nn.functional as F
from lithology_classification_model import ConvKernelExtractor
from lithology_classification_model import SEBlock
from lithology_classification_model import PrototypicalNetwork
from lithology_classification_model import DynamicPseudoLabeling


class DSSMLS(nn.Module):
    def __init__(
        self, in_channels, num_classes, embedding_dim=64, kernel_sizes=[3, 5], gamma=1.0
    ):
        """
        Initializes the Lithology Classification Model, combining Convolution Kernel Extractor,
        SE Block, Prototypical Network, and Dynamic Pseudo Labeling.

        :param in_channels: Number of input channels (features) for the convolution.
        :param num_classes: The number of classes for classification.
        :param embedding_dim: The embedding dimension for feature representation.
        :param kernel_sizes: List of kernel sizes for the convolution layers in the feature extractor.
        :param gamma: A parameter for controlling the sensitivity of pseudo-labeling.
        """
        super(DSSMLS, self).__init__()

        # Feature Extractor: ConvKernelExtractor followed by SEBlock
        self.feature_extractor = nn.Sequential(
            ConvKernelExtractor(in_channels, 128, kernel_sizes=kernel_sizes),
            SEBlock(128),  # Apply SEBlock for channel-wise attention
        )

        # Prototypical Network: To compute prototypes from support set and perform classification
        self.prototypical_network = PrototypicalNetwork(
            self.feature_extractor, num_classes, embedding_dim
        )

        # Dynamic Pseudo Labeling: To refine prototypes using unlabeled data
        self.dynamic_pseudo_labeling = DynamicPseudoLabeling(
            self.feature_extractor, num_classes, gamma=gamma
        )

        # Optional: Define a softmax layer if needed for classification output
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, support_set, query_set, unlabeled_set):
        """
        Forward pass through the model.
        This involves:
        1. Refining prototypes using dynamic pseudo-labeling.
        2. Using refined prototypes to classify query samples.

        :param support_set: Tensor of support set data, used to compute prototypes.
        :param query_set: Tensor of query set data, for which we need classification.
        :param unlabeled_set: Tensor of unlabeled samples, used for dynamic pseudo-labeling.
        :return: A tensor of distances between query set features and the prototypes.
        """
        # Step 1: Dynamic Pseudo-Labeling and Prototype Refinement
        refined_prototypes = self.dynamic_pseudo_labeling(
            support_set, unlabeled_set, query_set
        )

        # Step 2: Classification via Prototypical Network
        distances = self.prototypical_network(support_set, query_set)

        # Optionally: Apply softmax to distances for probability-like output
        probabilities = self.softmax(-distances)

        return distances, probabilities

    def calculate_loss(self, query_set, query_labels, distances):
        """
        Calculate the loss using negative log-likelihood loss function.

        :param query_set: The query set used for classification.
        :param query_labels: The ground truth labels for the query set.
        :param distances: The computed distances between query features and class prototypes.
        :return: The calculated loss value.
        """
        # Compute log probabilities
        log_p = F.log_softmax(-distances, dim=-1)

        # Negative log-likelihood loss (cross-entropy)
        loss = F.nll_loss(log_p, query_labels)

        return loss

    def update_prototypes(self, support_set, query_set, query_labels):
        """
        Update class prototypes based on support set and query set.

        :param support_set: The support set data (for few-shot learning).
        :param query_set: The query set data, which is used to test classification.
        :param query_labels: Ground truth labels of the query set.
        :return: Updated prototypes based on the support and query set.
        """
        support_features = self.feature_extractor(support_set)
        query_features = self.feature_extractor(query_set)

        # Compute prototypes from the support set (mean feature for each class)
        prototypes = self.prototypical_network.compute_prototypes(support_features)

        # Refine prototypes with query set features (e.g., weighted update or averaging)
        updated_prototypes = self.prototypical_network.compute_prototypes(
            query_features
        )

        return updated_prototypes

    def evaluate(self, query_set, query_labels, distances):
        """
        Evaluate the model's performance using accuracy.

        :param query_set: The query set used for classification.
        :param query_labels: The ground truth labels for the query set.
        :param distances: The distances between the query set and prototypes.
        :return: The classification accuracy of the model.
        """
        # Get the predicted labels by selecting the class with the smallest distance
        _, predicted_labels = torch.min(distances, dim=-1)

        # Calculate accuracy
        accuracy = (predicted_labels == query_labels).float().mean()

        return accuracy

    def save_model(self, save_path):
        """
        Save the model's state dictionary to a file.

        :param save_path: Path where the model will be saved.
        """
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """
        Load the model's state dictionary from a file.

        :param load_path: Path from where the model will be loaded.
        """
        self.load_state_dict(torch.load(load_path))
        print(f"Model loaded from {load_path}")
