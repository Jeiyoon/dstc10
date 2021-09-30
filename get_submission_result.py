import glob
import json
import numpy as np
import pandas as pd
import random
from scipy.stats import spearmanr
from collections import defaultdict
import argparse

def main(user_submission):
    # dev set direction/name
    f_list = glob.glob("dev_dataset/*.json")
    dialogue_ids = []
    dialogue_annotations = []
    dialogue_scores = []
    for f in f_list:
        d = json.load(open(f))
        #print(f)
        #compute annotation score in dev set
        for item in d:
            for key in item['annotations'].keys():
                if len(item['annotations'][key]) >= 1:
                    dialogue_ids.append(item['dialogue_id']+'|{}'.format(key))
                    dialogue_annotations.append(np.mean(item['annotations'][key]))
    #annotations scores = mean of annotations
    dict_1 = {1:dialogue_ids,2:dialogue_annotations}
    annotations_scores = pd.DataFrame(dict_1)
    annotations_scores.columns = ['dialogue_id', 'score']
    #print(len(user_submission))
    #print(len(annotations_scores))
    assert len(user_submission) == len(annotations_scores), "wrong number of entries in your submission"
    user_submission_dialogue_id = list(user_submission['dialogue_id'])
    annotation_dialogue_id = list(annotations_scores['dialogue_id'])
    user_submission_dialogue_id.sort()
    annotation_dialogue_id.sort()
    for x, y in zip(user_submission_dialogue_id, annotation_dialogue_id):
        assert x == y, "please check dialogue ids in your submission file"
    user_submitted_scores = list(user_submission['score'])
    '''
    for item in user_submitted_scores:
        if type(item) is not float or item < 0:
            print(item)
        assert type(item) is float and item >= 0
    '''
    #final_df = submission template 
    final_df = user_submission.sort_values('dialogue_id').reset_index(drop=True)
    annotations_df = annotations_scores.sort_values('dialogue_id').reset_index(drop=True)
    #put annotation_df into final_df-> annotation_score 이름으로
    final_df['annotation_score'] = annotations_df['score']
    dialogue_id_list = list(final_df['dialogue_id'])
    dataset_list = [item.split('_')[0] for item in dialogue_id_list]
    dimension_list = [item.split('|')[1] for item in dialogue_id_list]
    dataset_dimension_list =  [item.split('_')[0] + '|' + item.split('|')[1] for item in dialogue_id_list]
    final_df['dataset_list'] = dataset_list
    final_df['dimension_list'] = dimension_list
    final_df['dataset_dimension_list'] = dataset_dimension_list
    dataset_set = set(list(dataset_list))
    dataset_dimension_set = set(list(dataset_dimension_list))
    dataset_dimension_score = {}
    for item in dataset_dimension_set:
        temp_df = final_df[final_df['dataset_dimension_list']==item]
        #annotation_score(dev set annotation)과
        dataset_dimension_score[item] = spearmanr(temp_df['score'], temp_df['annotation_score'])[0]
    dataset_to_score = defaultdict(list)
    for k, v in dataset_dimension_score.items():
        d_name = k.split('|')[0]
        d_dimension = k.split('|')[1]
        dataset_to_score[d_name].append(v)
    dataset_to_avg_score = {k:np.mean(v) for k, v in dataset_to_score.items()}
    final_system_score = np.mean(list(dataset_to_avg_score.values()))
    results = {"score_by_dataset_dimension": dataset_dimension_score,
           "score_by_dataset": dataset_to_avg_score,
           "final_system_score": final_system_score}
    return results

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission_name', type=str, default='submission_template.csv')
    args = parser.parse_args()
    
    user_submission = pd.read_csv(args.submission_name)
    res = main(user_submission)
    print(res)
