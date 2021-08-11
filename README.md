# Track 5: Automatic Evaluation and Moderation of Open-domain Dialogue Systems 

## Members

- [Jeiyoon Park](http://jeiyoon.github.io/), Jieun Han

## Task Proposal and Track Website

- Task Proposal: https://drive.google.com/file/d/1B2YBtWaLJU5X3uudSZEaOyNWQ_QoTZLG/view
- Track Website: https://chateval.org/dstc10

## Baselines

### 1) Deep AM-FM: Toolkit for Automatic Dialogue Evaluation (Zhang et al., Lecture Notes in Electrical Engineering, vol 704. Springer, Singapore. 2021)

- Paper: https://link.springer.com/chapter/10.1007/978-981-15-8395-7_5 
- Code: https://github.com/e0397123/dstc10_metric_track/tree/main/baselines/deep_amfm

- Parameters (Fine-tuning AM on DSTC6)
```
--train_data_file=/root/dstc10/dstc10_metric_track-main/baselines/deep_amfm/twitter_trial_data_train_jeiyoon.txt
--output_dir=./embedding_models/full_am
--model_type=bert
--model_name_or_path=bert-base-uncased
--do_train
--do_eval
--eval_data_file=/root/dstc10/dstc10_metric_track-main/baselines/deep_amfm/DSTC_10_Track_5/Subtask_1/human_evaluation_data/human_evaluation_data/dstc6_eval.json
--overwrite_output_dir
--per_device_train_batch_size=4
--per_device_eval_batch_size=4
--block_size=512
--mlm
```


- Parameters (Fine-tuning FM on DSTC6)
```
--train_data_file=/root/dstc10/dstc10_metric_track-main/baselines/deep_amfm/twitter_trial_data_train_jeiyoon.txt
--output_dir=./language_models/full_fm
--model_type=gpt2
--model_name_or_path=gpt2
--do_train
--do_eval
--eval_data_file=/root/dstc10/dstc10_metric_track-main/baselines/deep_amfm/DSTC_10_Track_5/Subtask_1/human_evaluation_data/human_evaluation_data/dstc6_eval.json
--overwrite_output_dir
--per_device_train_batch_size=1
--per_device_eval_batch_size=1
--block_size=512
```


- Parameters (Fine-tuning AM on DSTC7)
```
--train_data_file=/root/dstc10/dstc10_metric_track-main/baselines/deep_amfm/reddit_train_jeiyoon.txt
--output_dir=./dstc7_model/embedding_models/full_am
--model_type=bert
--model_name_or_path=bert-base-uncased
--do_train
--do_eval
--eval_data_file=/root/dstc10/dstc10_metric_track-main/baselines/deep_amfm/DSTC_10_Track_5/Subtask_1/human_evaluation_data/human_evaluation_data/dstc7_eval.json
--overwrite_output_dir
--per_device_train_batch_size=4
--per_device_eval_batch_size=4
--block_size=512
--mlm
```


- Parameters (Fine-tuning FM on DSTC7)
```
--train_data_file=/root/dstc10/dstc10_metric_track-main/baselines/deep_amfm/reddit_train_jeiyoon.txt
--output_dir=./dstc7_model/language_models/full_fm
--model_type=gpt2
--model_name_or_path=gpt2
--do_train
--do_eval
--eval_data_file=/root/dstc10/dstc10_metric_track-main/baselines/deep_amfm/DSTC_10_Track_5/Subtask_1/human_evaluation_data/human_evaluation_data/dstc7_eval.json
--overwrite_output_dir
--per_device_train_batch_size=1
--per_device_eval_batch_size=1
--block_size=512
```


- Parameters (Compute Reference-based AM-FM Scores for Turn-level Dataset)

```
--dataset=dstc6 --device=cuda --am_model_path=embedding_models/full_am --fm_model_path=language_models/full_fm
```


```
[wr] DSTC6-Eval (D6) (Hori et al., 2017)
[wr] DSTC7-Eval (D7) (Galley et al., 2019)
[wr] DailyDialog-Eval (GD) (Gupta et al., 2019)
[wr] DailyDialog-Eval (ZD) (Zhao et al., 2020)
[wr] HUMOD (HU) (Merdivan et al., 2020)
[wr] PersonaChat-USR (UP) (Mehri & Eskenazi, 2020a)
[wr] PersonaChat-Eval (ZP) (Zhao et al., 2020)
[wr] TopicalChat-USR (TP) (Mehri & Eskenazi, 2020a)
```


- Parameters (Compute Reference-free AM-FM Scores for Turn-level Dataset)

```
--dataset=fed-turn --device=cuda:1 --am_model_path=embedding_models/full_am --fm_model_path=language_models/full_fm
```


```
[wor] FED-Turn (FT) (Mehri & Eskenazi, 2020b)
[wor] ConvAI2-Eval (EC) (Huang et al., 2020)
[wor] Empathetic-Eval (EE) (Huang et al., 2020)
[wor] DailyDialog-Eval (ED) (Huang et al., 2020)
```


- Parameters (Compute Reference-free AM-FM Scores for Dialogue-level Dataset)

```
--dataset=fed-dial --device=cuda:2 --am_model_path=embedding_models/full_am --fm_model_path=language_models/full_fm
```

```
[dial] FED-Conversation (FC) (Mehri & Eskenazi, 2020b)
[dial] Persona-Chatlog (PC) (See et al., 2019)
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

### 3) PersonaCHAT

TBA


## Deep Multi-Task and Meta-Learning

https://github.com/Jeiyoon/CS330-Stanford-Deep-Multi-Task-and-Meta-Learning

## Research Note (notion)

- https://www.notion.so/DSTC10-Automatic-Evaluation-and-Moderation-of-Open-domain-Dialogue-Systems-dc455b5598c240e3b4a20c66b5a884af

- The system-level correlation means that for a group of dialogue systems to rank, each system will receive a single metric score. Then we do correlation between the list of scores and the corresponding human annotated scores. In the experiment, one can simply average the scores of all the responses from a dialogue system in the test set and treat the averaged score as the system score. (same procedure for the corresponding human scores)

- Conversation-level correlation intends to rank a list of conversations. In the interactive human evaluation setting, the annotator will give a single rating to the entire conversation. Then the automatic metric will also need to assign a single score to the entire conversation. Correlation is performed between these two groups of scores. One simple way to obtain a single conversational level metric score is to average the scores assigned to all the context-response pairs within the conversation.

- Turn-level is the most fine-grained category. It is the common approach in static evaluation whereby we have multiple context-response pairs, the annotators will annotate the quality of the response. Then, the metric will assign score for each context-response pair. Correlation is performed between these two groups of scores
