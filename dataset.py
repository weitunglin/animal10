import os
from torch.utils.data import Dataset
from PIL import Image

class Animal10Dataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.transform = transform
        self.mapping = {"Dog": 4, "Horse": 6, "Elephant": 5, "Butterfly": 0, "Chicken": 2, "Cat": 1, "Cow": 3, "Sheep": 7, "Squirrel": 9, "Spider": 8}
  
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        return image, self.mapping[row["label"]]
  
    def __len__(self):
        return len(self.df)