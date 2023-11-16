from torch.utils.data import DataLoader
from .dataset import MyDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def getDataloder(dataset_name, n_skill, max_len=200, num_workers=1, batch_size=16):
  print("train on %s......"%dataset_name)
  current_dir = os.path.dirname(os.path.abspath(__file__))
  raw_path = "./processed/%s/%s_processed.csv"%(dataset_name, dataset_name)
  train_path = "./processed/%s/%s_train.csv"%(dataset_name, dataset_name)
  test_path = "./processed/%s/%s_test.csv"%(dataset_name, dataset_name)
  raw_path = os.path.join(current_dir, raw_path)
  train_path = os.path.join(current_dir, train_path)
  test_path = os.path.join(current_dir, test_path)

  raw = pd.read_csv(raw_path, sep=',')
  raw_shuffled = raw.sample(frac=1, random_state=42)
  train_data, test_data = train_test_split(raw_shuffled, test_size=0.2, random_state=42)
  train_data.reset_index(drop=True)
  test_data.reset_index(drop=True)
  train_data.to_csv(train_path, index=False)
  test_data.to_csv(test_path, index=False)

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
  print("data ready!")
  return train_dataloader, test_dataloader
