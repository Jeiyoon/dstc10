import glob
import json
import pandas as pd
import random

if __name__=='__main__':
    f_list = glob.glob("../dstc10-task5.1_eval.json")
    dialogue_ids = []
    dialogue_scores = []
    for f in f_list:
        d = json.load(open(f))
        for item in d:
            dialogue_ids.append(item['dialogue_id'])
            #random = placeholder
            dialogue_scores.append(random.random())
    dict_2 = {1:dialogue_ids,2:dialogue_scores}
    dummy_df = pd.DataFrame(dict_2)
    dummy_df.columns = ['dialogue_id', 'score']
    # dummy submission csv file
    dummy_df.to_csv("submission_template_test.csv", index=None)

