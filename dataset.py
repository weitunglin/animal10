import os
from torch.utils.data import Dataset
from PIL import Image

class Animal10Dataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.transform = transform
        self.mapping = {"Dog": 0, "Horse": 1, "Elephant": 2, "Butterfly": 3, "Chicken": 4, "Cat": 5, "Cow": 6, "Sheep": 7, "Squirrel": 8, "Spider": 9}
  
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        return image, self.mapping[row["label"]]
  
    def __len__(self):
        return len(self.df)