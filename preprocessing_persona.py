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
# print(df)
df2 = df[['UID', 'SID', 'SEG', 'CAND']]
# print(df2)
# df3 = df['SEG']
# print(df[['UID', 'SID', 'SEG']])
#
context_idx = ""
discarded_utt = 0
# max_context_length = -sys.maxsize
# context_length = 0 # max context length = 8
for idx, (u, s, g, c) in enumerate(zip(df['UID'], df['SID'], df['SEG'], df['CAND'])):
    # reddit-000002-0000 -> [reddit,000002,0000]
    uid_split_list = u.split('-')

    # if uid_split_list[1] == "000005":
    #     break

    # print(uid_split_list)
    # print("uid:", uid_split_list[1])
    # cand_split_list = c.split('|')

    if len(uid_split_list) == 3:
        if s == 'partner' or s == 'human_evaluator': # U
            s = 'U: '
        elif s == 'self' or s == 'model':
            s = 'S: '
        else:
            print("ValueError")

        # for c_idx, cand in enumerate(cand_split_list):
        #     print("CAND_" + str(c_idx) + ": " + cand)

        if context_idx != uid_split_list[1]:
            print(" ")
            persona.write("\n")
            context_idx = uid_split_list[1]
        else:
            pass

        utterance = s + g
        print(utterance)
        persona.write(utterance)
        persona.write("\n")
    else:
        pass # evalset -> discard

persona.close()

