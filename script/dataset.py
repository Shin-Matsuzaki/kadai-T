from pathlib import Path, PosixPath

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class myDataset(Dataset):
    def __init__(self, data_path: PosixPath, transform):
        self.file_path_list = list(data_path.glob('**/*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, idx):
        file_path = self.file_path_list[idx]
        image = Image.open(file_path)
        image_transformed = self.transform(image)

        # label:アリ=>0 / ハチ=>1
        dir_name = file_path.parent.name
        if dir_name == 'ants':
            label = 0
        elif dir_name == 'bees':
            label = 1
        else:
            raise ValueError(f'{dir_name} is invalid')
        return image_transformed, label


def main():
    train_path = Path('../hymenoptera_data/train')
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = myDataset(train_path, train_transform)
    valid_path = Path('../hymenoptera_data/val')
    valid_dataset = myDataset(valid_path, train_transform)

    print(train_dataset[121][1])


if __name__ == '__main__':
    main()