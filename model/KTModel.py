'''
Created: 2023/9/19
'''
import torch
import torch.nn as nn

class KTModel(nn.Module):
  def __init__(self, n_skill, n_diff, n_hints, n_gap, n_time_used, max_len, emb_dim, bs):
    super().__init__()
    self.max_len = max_len
    self.emb_dim = emb_dim

    self.skill_emb = nn.Embedding(n_skill+2, emb_dim, padding_idx=n_skill+1)
    self.answer_emb = nn.Embedding(3, emb_dim, padding_idx=2)
    self.diff_emb = nn.Embedding(n_diff+2, emb_dim, padding_idx=n_diff+1)
    self.hints_emb = nn.Embedding(n_hints+2, emb_dim, padding_idx=n_hints+1)
    self.gap_emb = nn.Embedding(n_gap+2, emb_dim, padding_idx=n_gap+1)
    self.time_used_emb = nn.Embedding(n_time_used+2, emb_dim, padding_idx=n_time_used+1)
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.bs = bs

    self.W_q_1 = nn.Parameter(torch.Tensor(bs, self.emb_dim))
    nn.init.xavier_uniform_(self.W_q_1)
    self.W_k_1 = nn.Parameter(torch.Tensor(bs, self.emb_dim))
    nn.init.xavier_uniform_(self.W_k_1)
    self.W_v_1 = nn.Parameter(torch.Tensor(bs, self.emb_dim))
    nn.init.xavier_uniform_(self.W_v_1)
 
    self.W_q_2 = nn.Parameter(torch.Tensor(bs, self.emb_dim))
    nn.init.xavier_uniform_(self.W_q_2)
    self.W_k_2 = nn.Parameter(torch.Tensor(bs, self.emb_dim))
    nn.init.xavier_uniform_(self.W_k_2)
    self.W_v_2 = nn.Parameter(torch.Tensor(bs, self.emb_dim))
    nn.init.xavier_uniform_(self.W_v_2)

    self.W_q_3 = nn.Parameter(torch.Tensor(bs, self.emb_dim))
    nn.init.xavier_uniform_(self.W_q_3)
    self.W_k_3 = nn.Parameter(torch.Tensor(bs, self.emb_dim))
    nn.init.xavier_uniform_(self.W_k_3)
    self.W_v_3 = nn.Parameter(torch.Tensor(bs, self.emb_dim))
    nn.init.xavier_uniform_(self.W_v_3)

    self.fc = nn.Linear(2*emb_dim, emb_dim)
    self.fc2 = nn.Linear(2*emb_dim, emb_dim)

  def forward(self, skill, answer, diff, hints, gap, time_used):
    # input: (batch_size, seq_len, emb_dim)
    embed_skill = self.skill_emb(skill)
    embed_diff = self.diff_emb(diff)
    embed_a = self.answer_emb(answer)
    embed_hints = self.hints_emb(hints)
    embed_gap = self.gap_emb(gap)
    embed_time_used = self.time_used_emb(time_used)

    x = torch.cat((embed_skill, embed_diff), 2)
    x = self.fc(x)
    assert x.size(2) == self.emb_dim

    # TODO: 用randn还是zeros？
    # h,c: batch_size, emb_dim
    h_t, c_t = (torch.zeros(self.bs, self.emb_dim).to(self.device),
                torch.zeros(self.bs, self.emb_dim).to(self.device))

    y_preds = []
    y_preds.append(torch.zeros(self.bs))

    for t in range(self.max_len-1):
      x_t = x[:, t, :] # batch_size, emb_dim
      x_next = x[:, t+1, :] # batch_size, emb_dim
      # forget
      gap_t = embed_gap[:, t, :]
      gap_t = torch.sigmoid(gap_t) # 0~1
      h_t = h_t - gap_t

      # 3 user responses
      used_t = embed_time_used[:, t, :]
      hints_t = embed_hints[:, t, :]
      a_t = embed_a[:, t, :]

      x_t = self.fc2(torch.cat((x_t, h_t), 1))

      q_1 = self.W_q_1 * x_t
      k_1 = self.W_k_1 * c_t
      v_1 = self.W_v_1 * used_t
      score_1 = torch.sigmoid(q_1 * k_1 / torch.sqrt(torch.ones(self.bs, self.emb_dim) * self.emb_dim))
      out_1 = v_1 * score_1

      q_2 = self.W_q_2 * x_t
      k_2 = self.W_k_2 * c_t
      v_2 = self.W_v_2 * hints_t
      score_2 = torch.sigmoid(q_2 * k_2 / torch.sqrt(torch.ones(self.bs, self.emb_dim) * self.emb_dim))
      out_2 = v_2 * score_2

      # fusion gate 1
      out = out_1 + out_2
      c_t = torch.tanh(out) * torch.sigmoid(out) + c_t

      q_3 = self.W_k_3 * x_t
      k_3 = self.W_q_3 * h_t
      v_3 = self.W_v_3 * a_t
      score_3 = torch.sigmoid(q_3 * k_3 / torch.sqrt(torch.ones(self.bs, self.emb_dim) * self.emb_dim))
      out_3 = v_3 * score_3

      # fusion gate 2
      h_t = torch.tanh(out_3) * torch.sigmoid(out_3) + c_t
      ccc=h_t.size()

      _y_pred = torch.sigmoid(torch.sum(x_next * h_t, dim=1))
      y_preds.append(_y_pred)

    y_pred = torch.stack(y_preds, dim=1)
    return y_pred