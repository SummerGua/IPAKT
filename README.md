# Improving Knowledge Tracing via Considering Students' Interaction Patterns

Pytorch Implementation

## Architecture
![](./imgs/model.svg)

## Usage
put processed data under `./data/processed/assist[09/12/17]/`

data file name: `assist[09/12/17]_processed.csv`

processed data format:
```csv
user_id,skill_id,correct,ms_first_response,hint_count,difficulty
70657,"8,8,3,3","0,0,0,0","63,24,63,24","0,0,0,0","0.84,0.66,0.84,0.66"
```

run:

- `python main.py --dataset assist[09/12/17]`