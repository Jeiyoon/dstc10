import glob
import json
import pandas as pd
import random

if __name__=='__main__':
    f_list = glob.glob("dev_dataset/*.json")
    dialogue_ids = []
    dialogue_scores = []
    for f in f_list:
        d = json.load(open(f))
        result_csv = f.split("\\")[1].split("eval")[0] + "results.csv"
        path = "result_maml/" + result_csv
        result_df = pd.read_csv(path, encoding = 'ISO-8859-1')

        for i in range(len(d)):
            item = d[i]
            for key in item['annotations'].keys():
                if len(item['annotations'][key]) >= 1:
                    name = "annotations." + key
                    id = item['dialogue_id']+'|{}'.format(key)
                    dialogue_ids.append(item['dialogue_id']+'|{}'.format(key))
                    dialogue_scores.append(result_df[name].values[i])


    dict_2 = {1:dialogue_ids,2:dialogue_scores}
    dummy_df = pd.DataFrame(dict_2)
    dummy_df.columns = ['dialogue_id', 'score']
    # dummy submission csv file
    dummy_df.to_csv("submission_template.csv", index=None)

