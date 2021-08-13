# author: Jeiyoon
import pandas as pd

"""
/////////////////////////
YOU SHOULD CHECK THIS OUT
/////////////////////////
"""
data_path = "/root/dstc10/dstc10_metric_track-main/baselines/deep_amfm/ds_dataset/persona_main.csv"
persona = open("/root/dstc10/dstc10_metric_track-main/baselines/deep_amfm/persona_train_jeiyoon.txt", 'w')

# idx 317984 -> evalset -> discard

# Unnamed
# UID: persona-000000-0
# UID: persona-usr-seq2seq-000007-0000

# SID: partner or self
# SID: human_evaluator or model

# SEG: great ! i will bring this up to her tomorrow .

# CAND(if SID is partner): _empty
# CAND(if SID is self): no it is not . i ' m scared a night and i used to like nighttime a lot .|i bet you are very little|i enjoy jazz and r and b|they sure are . we do ! there are 8 girls and 2 boys .|hi how are you doing ?|hello , my names mary . how are you doing today ?|i should , but if i could i lose my mind too .|are you broke l o l ?|vegas is fun to visit but not to live there|i had a simple lunch so that i make my supper heavy after beach .|that ' s awesome you get to do that . you get to enjoy the kids more now|you work for a funeral home ?|oh damn i ' m sorry to hear that|i draw for a living .|paper , sales , its what i do , salesman , also beets .|hi my name is spongebob how are you|what are your hobbies i prefer to read alot|marketing i enjoy it really|haha . i don ' t see the link|i ' ve over 40 million dollars

df = pd.read_csv(data_path)
context_idx = ""

for idx, (u, s, g, c) in enumerate(zip(df['UID'], df['SID'], df['SEG'], df['CAND'])):
    uid_split_list = u.split('-')

    if len(uid_split_list) == 3:
        if s == 'partner' or s == 'human_evaluator': # U
            s = 'U: '
        elif s == 'self' or s == 'model':
            s = 'S: '
        else:
            print("ValueError")

        if context_idx != uid_split_list[1]:
            persona.write("\n")
            context_idx = uid_split_list[1]
        else:
            pass

        utterance = s + g
        persona.write(utterance)
        persona.write("\n")
    else:
        pass # evalset -> discard

persona.close()
