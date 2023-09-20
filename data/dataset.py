import torch
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
  def __init__(self, dataframe, n_skill, max_len=100):
    self.df = dataframe
    self.n_skill = n_skill
    self.max_len = max_len
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, index):
    '''
    # NOTICE
    - length of sequence here should be < max_len
    - each time_used should be <= 300
    '''
    skill_id         = self.df['concept'][index].split(',')
    difficulty       = self.df['difficulty'][index].split(',')
    gap_time         = self.df['gap_time'][index].split(',')
    answer           = self.df['answer'][index].split(',')
    time_used        = self.df['time_used'][index].split(',')
    hints            = self.df['hints'][index].split(',')

    skill_id         = np.array(list(map(int,   skill_id)))
    difficulty_level = np.array(list(map(float, difficulty))) * 100
    difficulty_level = difficulty_level.astype(int)
    gap_time         = np.array(list(map(int,   gap_time)))
    answer           = np.array(list(map(int,   answer)))
    time_used        = np.array(list(map(int,   time_used)))
    hints            = np.array(list(map(int,   hints)))

    assert len(answer) == len(time_used) == len(gap_time) == \
           len(difficulty_level) == len(skill_id) == len(hints)

    current_len = len(answer) # actual length of current sequence

    truth = np.ones(self.max_len, dtype=int) * -1
    truth[:current_len] = answer

    # mask the predicted values
    mask = np.zeros(self.max_len, dtype=int)
    mask[:current_len] = 1 # take 1st~len-1 prediction
    mask[0] = 0.5 # take 2nd~last truth

    q = np.ones(self.max_len, dtype=int) * (self.n_skill + 1)
    q[:current_len] = skill_id

    a = np.ones(self.max_len, dtype=int) * 2
    a[:current_len] = answer

    t_used = np.ones(self.max_len, dtype=int) * 301
    t_used[:current_len] = time_used

    diff_level = np.ones(self.max_len, dtype=int) * 101
    diff_level[:current_len] = difficulty_level

    gap = np.ones(self.max_len, dtype=int) * 301
    gap[:current_len] = gap_time

    n_hints = np.ones(self.max_len, dtype=int) * 11
    n_hints[:current_len] = hints

    return (
      torch.LongTensor(q),
      torch.LongTensor(diff_level),
      torch.LongTensor(a),
      torch.LongTensor(t_used),
      torch.LongTensor(n_hints),
      torch.LongTensor(gap),
      torch.LongTensor(truth),
      torch.LongTensor(mask),
    )