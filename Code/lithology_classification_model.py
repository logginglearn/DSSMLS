import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvKernelExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5]):
        super(ConvKernelExtractor, self).__init__()
        self.kernel_sizes = kernel_sizes

        # Convolution layers for different kernels
        self.conv_3 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2,
        )
        self.conv_5 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2,
        )
        self.bn_3 = nn.BatchNorm1d(out_channels)
        self.bn_5 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # Apply convolutions with different kernels
        x_3 = F.relu(self.bn_3(self.conv_3(x)))
        x_5 = F.relu(self.bn_5(self.conv_5(x)))

        # Combine outputs
        x_combined = x_3 + x_5
        return x_combined


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Average Pooling
        avg_pool = torch.mean(x, dim=-1)  # Shape: [batch_size, channels]

        # Excitation
        excitation = F.relu(self.fc1(avg_pool))
        excitation = self.sigmoid(self.fc2(excitation)).unsqueeze(-1)

        # Scaling
        return x * excitation


class PrototypicalNetwork(nn.Module):
    def __init__(self, feature_extractor, num_classes, embedding_dim=64):
        super(PrototypicalNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

    def forward(self, support_set, query_set):
        # Extract features for support and query sets
        support_features = self.feature_extractor(support_set)
        query_features = self.feature_extractor(query_set)

        # Compute prototypes (mean of support set features per class)
        prototypes = self.compute_prototypes(support_features)

        # Calculate distances from query samples to prototypes
        distances = self.compute_distances(query_features, prototypes)

        return distances

    def compute_prototypes(self, support_features):
        # Calculate class prototypes (mean of support features for each class)
        prototypes = []
        for c in range(self.num_classes):
            class_features = support_features[
                support_features[:, -1] == c, :-1
            ]  # Last column is label
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes)

    def compute_distances(self, query_features, prototypes):
        # Compute the Euclidean distance between query features and prototypes
        distances = torch.cdist(query_features, prototypes)
        return distances

    def calculate_loss(self, query_set, query_labels, distances):
        # Negative log-likelihood loss
        log_p = F.log_softmax(-distances, dim=-1)
        loss = F.nll_loss(log_p, query_labels)
        return loss


class DynamicPseudoLabeling(nn.Module):
    def __init__(self, feature_extractor, num_classes, gamma=1.0):
        super(DynamicPseudoLabeling, self).__init__()
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.gamma = gamma

    def forward(self, support_set, unlabeled_set, labeled_set):
        # Extract features
        support_features = self.feature_extractor(support_set)
        unlabeled_features = self.feature_extractor(unlabeled_set)

        # Generate pseudo-labels for unlabeled data
        pseudo_labels = self.generate_pseudo_labels(
            unlabeled_features, support_features
        )

        # Update prototypes using labeled and pseudo-labeled samples
        refined_prototypes = self.update_prototypes(
            support_features, pseudo_labels, labeled_set
        )

        return refined_prototypes

    def generate_pseudo_labels(self, unlabeled_features, support_features):
        # Compute distances from unlabeled samples to support class prototypes
        prototypes = self.compute_prototypes(support_features)
        distances = torch.cdist(unlabeled_features, prototypes)

        # Apply tunable distance factor (gamma) for weighting
        pseudo_labels = torch.exp(-self.gamma * distances)

        return pseudo_labels

    def update_prototypes(self, support_features, pseudo_labels, labeled_set):
        # Weighted combination of labeled and pseudo-labeled features
        updated_prototypes = []
        for c in range(self.num_classes):
            # Get labeled features for class c
            labeled_class_features = support_features[support_set[:, -1] == c]

            # Get pseudo-labeled features for class c
            pseudo_class_features = pseudo_labels[:, c] * support_features

            # Combine labeled and pseudo-labeled features
            combined_features = torch.cat(
                [labeled_class_features, pseudo_class_features]
            )
            updated_prototypes.append(combined_features.mean(dim=0))

        return torch.stack(updated_prototypes)


 