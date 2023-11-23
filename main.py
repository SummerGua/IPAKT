import torch
from data.dataloader import getDataloder
from model import KTModel, KTLoss
import argparse
from tqdm import tqdm
import json
import utils
import os
import datetime

def run(args):
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # device = torch.device('cpu')

  # load config
  dataset_name = args.dataset

  current_dir = os.path.dirname(os.path.abspath(__file__))

  with open(os.path.join(current_dir, "./config.json"), "r") as config_json:
    config = json.load(config_json)
  print("config loaded!")
  batch_size = config[dataset_name]["batch_size"]
  max_epoch = config[dataset_name]["max_epoch"]
  max_len = config[dataset_name]["max_len"]
  n_skill = config[dataset_name]["n_skill"]
  num_workers = config[dataset_name]["num_workers"]
  lr = config[dataset_name]["learning_rate"]
  emb_dim = config[dataset_name]["emb_dim"]

  now_time = datetime.datetime.now()
  with open('./checkpoint/%s/performance_record.txt'%dataset_name, 'a+') as f:
        f.write("bs=%s, dim=%s, num_workers=%s\n%s\n" % (str(batch_size), str(emb_dim), str(num_workers), str(now_time)))
  
  # load data
  train_dataloader, test_dataloader = getDataloder(dataset_name, n_skill, max_len, num_workers, batch_size)

  # load model and optimizer
  model = KTModel(n_skill, 100, 10, 300, max_len, emb_dim)
  model.to(device)
  loss_fun = KTLoss()
  loss_fun.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
  print("model ready!")
  # train
  for epoch in tqdm(range(max_epoch)):
    utils.train_one_epoch(model, train_dataloader, optimizer, loss_fun, device)
    utils.eval_one_epoch(model, test_dataloader, device, dataset_name, epoch)
    if epoch >= 100 and (epoch+1) % 5 == 0:
      checkpoint = {
        'model': model.state_dict(),
        'potimizer': optimizer.state_dict(),
        'epoch': epoch
      }
      if dataset_name != "mock":
        torch.save(checkpoint,'./checkpoint/%s/bs%sdim%slen%s-epoch%s.pth' % (dataset_name, str(batch_size), str(emb_dim), str(max_len), str(epoch+1)))
    scheduler.step()

if __name__ == "__main__":
  print("started...")
  arg_parser = argparse.ArgumentParser(description="Ready to train...")
  arg_parser.add_argument("--dataset",
                          dest="dataset",
                          default="assist09",
                          type=str,
                          required=False)
  args = arg_parser.parse_args()
  run(args)