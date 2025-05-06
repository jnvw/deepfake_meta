import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        self.image_paths = []
        self.labels = []

        if mode == 'train':
            # For training, use the 'training_real' and 'training_fake' folders
            for label, subfolder in enumerate(['training_real', 'training_fake']):
                folder_path = os.path.join(root_dir, subfolder)
                for file in os.listdir(folder_path):
                    if file.endswith(('.jpg', '.png')):
                        self.image_paths.append(os.path.join(folder_path, file))
                        self.labels.append(label)
        
        elif mode == 'test':
            # For testing, use 'test_real' and 'test_fake' folders
            for label, subfolder in enumerate(['test_real', 'test_fake']):
                folder_path = os.path.join(root_dir, subfolder)
                for file in os.listdir(folder_path):
                    if file.endswith(('.jpg', '.png')):
                        self.image_paths.append(os.path.join(folder_path, file))
                        self.labels.append(label)
        else:
            raise ValueError("Mode must be 'train' or 'test'")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(image), self.labels[idx]
