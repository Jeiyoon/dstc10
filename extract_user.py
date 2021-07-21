# author: Jeiyoon
from typing import List
from glob import glob

# print(glob('/root/DSTC6-End-to-End-Conversation-Modeling/collect_twitter_dialogs/stored_data/*'))
# print(len(glob('/root/DSTC6-End-to-End-Conversation-Modeling/collect_twitter_dialogs/stored_data/*')))

root = "/root/DSTC6-End-to-End-Conversation-Modeling/collect_twitter_dialogs/stored_data/"
users = []

files: List[str] = glob('/root/DSTC6-End-to-End-Conversation-Modeling/collect_twitter_dialogs/stored_data/*')

for f in files:
    f = f.replace(root, "")
    f = f.replace(".json", "")
    users.append(f)

account_names = open("/root/dstc10/dstc10_metric_track-main/baselines/deep_amfm/users.txt", 'w')

for u in sorted(users):
    print(u)
    account_names.write(u+"\n")

account_names.close()
