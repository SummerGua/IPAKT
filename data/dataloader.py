from torch.utils.data import DataLoader
from .dataset import MyDataset
import pandas as pd
import os

def getDataloder(dataset_name, n_skill, max_len=200, num_workers=1, batch_size=16):
  current_dir = os.path.dirname(os.path.abspath(__file__))
  train_path = "./processed/" + dataset_name + "_train.csv"
  test_path = "./processed/" + dataset_name + "_test.csv"
  train_path = os.path.join(current_dir, train_path)
  test_path = os.path.join(current_dir, test_path)

  train_df = pd.read_csv(train_path, sep=',')
  test_df = pd.read_csv(test_path, sep=',')

  train_dataset = MyDataset(train_df, n_skill, max_len)
  test_dataset = MyDataset(test_df, n_skill, max_len)

  train_dataloader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True,
                                drop_last=True)
  test_dataloader = DataLoader(test_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False,
                                drop_last=True)

  return train_dataloader, test_dataloader
