import torch
from data.dataloader import getDataloder
from model import KTModel, KTLoss
import argparse
from tqdm import tqdm
import json
import utils
import os

def run(args):
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # load config
  dataset_name = args.dataset
  current_dir = os.path.dirname(os.path.abspath(__file__))
  with open(os.path.join(current_dir, "./config.json"), "r") as config_json:
    config = json.load(config_json)

  batch_size = config[dataset_name]["batch_size"]
  max_epoch = config[dataset_name]["max_epoch"]
  max_len = config[dataset_name]["max_len"]
  n_skill = config[dataset_name]["n_skill"]
  num_workers = config[dataset_name]["num_workers"]
  lr = config[dataset_name]["learning_rate"]
  emb_dim = config[dataset_name]["emb_dim"]
  
  # load data
  train_dataloader, test_dataloader = getDataloder(dataset_name, n_skill, max_len, num_workers, batch_size)

  # load model and optimizer
  model = KTModel(n_skill, 100, 10, 300, 300, max_len, emb_dim, batch_size)
  loss_fun = KTLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

  # train
  for epoch in tqdm(range(max_epoch)):
    utils.train_one_epoch(model, train_dataloader, optimizer, loss_fun, device)
    utils.eval_one_epoch(model, test_dataloader, device)
    scheduler.step()

if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser(description="Ready to train...")
  arg_parser.add_argument("--dataset",
                          dest="dataset",
                          default="mock",
                          type=str,
                          required=False)
  args = arg_parser.parse_args()
  run(args)