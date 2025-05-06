import random
import torch

def create_episode(dataset, n_way=2, k_shot=5, k_query=5):
    support_images, support_labels = [], []
    query_images, query_labels = [], []

    class_indices = {label: [] for _, label in dataset}
    for i, (_, label) in enumerate(dataset):
        class_indices[label].append(i)

    # Sort the dictionary keys before sampling
    selected_classes = random.sample(sorted(class_indices.keys()), n_way)
    label_map = {cls: i for i, cls in enumerate(selected_classes)}

    for cls in selected_classes:
        idxs = random.sample(class_indices[cls], k_shot + k_query)
        for i in idxs[:k_shot]:
            img, _ = dataset[i]
            support_images.append(img)
            support_labels.append(label_map[cls])
        for i in idxs[k_shot:]:
            img, _ = dataset[i]
            query_images.append(img)
            query_labels.append(label_map[cls])

    return torch.stack(support_images), torch.tensor(support_labels), torch.stack(query_images), torch.tensor(query_labels)
