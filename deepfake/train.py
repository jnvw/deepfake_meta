# Training script using episodic training for zero-shot learning
import torch
from data.dataset import DeepfakeDataset
from models.encoder import ResNetEncoder
from models.metaoptnet_svm import MetaOptNetSVM
from utils.few_shot_sampler import create_episode

train_dataset = DeepfakeDataset("data")
encoder = ResNetEncoder()
classifier = MetaOptNetSVM()

# If you only want to train the encoder (optional), you can use optimizer:
# optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

for episode in range(11):
    support_x, support_y, query_x, query_y = create_episode(train_dataset)
    support_features = encoder(support_x)
    query_features = encoder(query_x)

    preds = classifier(support_features, support_y, query_features)
    
    # preds shape: (batch_size, num_classes), query_y: (batch_size,)
    loss = torch.nn.CrossEntropyLoss()(preds, query_y)
    if episode % 4 == 0:
        torch.save(encoder.state_dict(), f"models/encoder_ep{episode}.pth")


    # Skip backward() and optimizer.step() because SVM is non-differentiable
    print(f"Episode {episode}, Loss: {loss.item():.4f}")
