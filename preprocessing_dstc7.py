# author: Jeiyoon
import sys
import pandas as pd

"""
/////////////////////////
YOU SHOULD CHECK THIS OUT
/////////////////////////
"""
data_path = "/root/dstc10/dstc10_metric_track-main/baselines/deep_amfm/ds_dataset/dstc7_main.csv"
dstc7 = open("/root/dstc10/dstc10_metric_track-main/baselines/deep_amfm/reddit_train_jeiyoon.txt", 'w')

# Unnamed
# UID: reddit-000002-0000
# SID: speaker_A
# SEG: A,til three minutes finale
df = pd.read_csv(data_path)

context_idx = ""
discarded_utt = 0

for idx, (u, s, g) in enumerate(zip(df['UID'], df['SID'], df['SEG'])):
    # reddit-000002-0000 -> [reddit,000002,0000]
    uid_split_list = u.split('-')

    if len(uid_split_list) == 3:
        # speaker_X -> U or S
        if s == 'speaker_A':
            s = 'U: '
        else:
            s = 'S: '

        if context_idx != uid_split_list[1]:
            print(" ")
            dstc7.write("\n")
            context_idx = uid_split_list[1]
        else:
            pass

        utterance = s + g
        dstc7.write(utterance)
        dstc7.write("\n")

    else:
        discarded_utt += 1
