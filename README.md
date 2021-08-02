# Track 5: Automatic Evaluation and Moderation of Open-domain Dialogue Systems 

## Task Proposal and Track Website

- Task Proposal: https://drive.google.com/file/d/1B2YBtWaLJU5X3uudSZEaOyNWQ_QoTZLG/view
- Track Website: https://chateval.org/dstc10

## Baselines

### 1) Deep AM-FM: Toolkit for Automatic Dialogue Evaluation (Zhang et al., Lecture Notes in Electrical Engineering, vol 704. Springer, Singapore. 2021)

- Paper: https://link.springer.com/chapter/10.1007/978-981-15-8395-7_5 
- Code: https://github.com/e0397123/dstc10_metric_track/tree/main/baselines/deep_amfm

- Parameters
```
--train_data_file=/root/dstc10/dstc10_metric_track-main/baselines/deep_amfm/twitter_trial_data_train_jeiyoon.txt
\
--output_dir=embedding_models/full_am
\
--model_type=bert
\
--model_name_or_path=bert-base-uncased
\
--do_train=True
\
--do_eval=True
\
--eval_data_file=/root/dstc10/dstc10_metric_track-main/baselines/deep_amfm/DSTC_10_Track_5/Subtask_1/human_evaluation_data/human_evaluation_data/dstc6_eval.json
\
--overwrite_output_dir
\
--per_device_train_batch_size=4
\
--per_device_eval_batch_size=4
\
--block_size=512
\
--mlm
```

### 2) D-score: Holistic Dialogue Evaluation without Reference (Zhang et al., IEEE/ACM Transactions on Audio, Speech, and Language Processing. 2021)

- Paper: https://ieeexplore.ieee.org/document/9409633
- Code: https://github.com/e0397123/D-score

## Datasets

### 1) DSTC6 Customer Support Dataset & The Human Evaluation Dataset

https://github.com/dialogtekgeek/DSTC6-End-to-End-Conversation-Modeling

For any use of the dataset, please cite
```
@article{hori2017end,
  title={End-to-end conversation modeling track in DSTC6},
  author={Hori, Chiori and Hori, Takaaki},
  journal={arXiv preprint arXiv:1706.07440},
  year={2017}
}
```

### 2) DSTC7 Knowledge Grounding Dataset & The Human Evaluation Dataset

https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling

For any use of the dataset, please cite
```
@inproceedings{Galley2019GroundedRG,
  title={Grounded Response Generation Task at DSTC7},
  author={Michel Galley and Chris Brockett and Xiang Gao and Jianfeng Gao and B. Dolan},
  booktitle = {Dialog System Technology Challenges (DSTC7)},
  year={2019}
}
```

## Deep Multi-Task and Meta-Learning

https://github.com/Jeiyoon/CS330-Stanford-Deep-Multi-Task-and-Meta-Learning

