import os
import json
import time
import numpy as np

NUM_RUNS = 5
res_dict_1K = {('de', 'fi'): {'eps': 0.05,  'lamb': 1.25,  'paper_mrr': 0.345},
               ('de', 'fr'): {'eps': 0.05,  'lamb': 1.25,  'paper_mrr': 0.530},
               ('de', 'hr'): {'eps': 0.1,   'lamb': 1.0,   'paper_mrr': 0.312},
               ('de', 'it'): {'eps': 0.01,  'lamb': 0.25,  'paper_mrr': 0.526},
               ('de', 'ru'): {'eps': 0.1,   'lamb': 1.0,   'paper_mrr': 0.368},
               ('de', 'tr'): {'eps': 0.1,   'lamb': 0.75,  'paper_mrr': 0.275},
               ('en', 'de'): {'eps': 0.05,  'lamb': 0.75,  'paper_mrr': 0.568},
               ('en', 'fi'): {'eps': 0.025, 'lamb': 1.25,  'paper_mrr': 0.397},
               ('en', 'fr'): {'eps': 0.1,   'lamb': 0.5,   'paper_mrr': 0.666},
               ('en', 'hr'): {'eps': 0.05,  'lamb': 1.25,  'paper_mrr': 0.320},
               ('en', 'it'): {'eps': 0.05,  'lamb': 1.0,   'paper_mrr': 0.638},
               ('en', 'ru'): {'eps': 0.05,  'lamb': 1.25,  'paper_mrr': 0.439},
               ('en', 'tr'): {'eps': 0.1,   'lamb': 0.5,   'paper_mrr': 0.360},
               ('fi', 'fr'): {'eps': 0.1,   'lamb': 1.25,  'paper_mrr': 0.366}, 
               ('fi', 'hr'): {'eps': 0.05,  'lamb': 1.25,  'paper_mrr': 0.304},
               ('fi', 'it'): {'eps': 0.15,  'lamb': 0.75,  'paper_mrr': 0.372},
               ('fi', 'ru'): {'eps': 0.1,   'lamb': 1.0,   'paper_mrr': 0.346},
               ('hr', 'fr'): {'eps': 0.1,   'lamb': 0.75,  'paper_mrr': 0.380},
               ('hr', 'it'): {'eps': 0.1,   'lamb': 1.25,  'paper_mrr': 0.389},
               ('hr', 'ru'): {'eps': 0.1,   'lamb': 0.5,   'paper_mrr': 0.380},
               ('it', 'fr'): {'eps': 0.025, 'lamb': 1.25,  'paper_mrr': 0.678},
               ('ru', 'fr'): {'eps': 0.05,  'lamb': 1.0,   'paper_mrr': 0.486},
               ('ru', 'it'): {'eps': 0.1,   'lamb': 0.75,  'paper_mrr': 0.489},
               ('tr', 'fi'): {'eps': 0.15,  'lamb': 1.25,  'paper_mrr': 0.280},
               ('tr', 'fr'): {'eps': 0.1,   'lamb': 1.25,  'paper_mrr': 0.342},
               ('tr', 'hr'): {'eps': 0.15,  'lamb': 0.5,   'paper_mrr': 0.241},
               ('tr', 'it'): {'eps': 0.1,   'lamb': 0.75,  'paper_mrr': 0.335},
               ('tr', 'ru'): {'eps': 0.1,   'lamb': 1.25,  'paper_mrr': 0.248}}

for (lang1, lang2) in res_dict_1K:
    res_dict_1K[(lang1, lang2)]['runtime'] = []
    for run_idx in range(NUM_RUNS):
        eps, lamb = res_dict_1K[lang1, lang2]['eps'], res_dict_1K[lang1, lang2]['lamb']

        ## Code to profile ##
        start_time = time.time()
        os.system('python code/map.py -m f -d bli_datasets/%s-%s/yacle.train.freq.1k.%s-%s.tsv --lang_src %s --lang_trg %s ft-raw-200k/vecs_%s ft-raw-200k/vocab_%s ft-raw-200k/vecs_%s ft-raw-200k/vocab_%s ft-raw-200k/ --eps %s --lamb %s --self_learn_num 14000'% (lang1, lang2, lang1, lang2, lang1, lang2, lang1, lang1, lang2, lang2, eps, lamb))
        res_dict_1K[(lang1, lang2)]['runtime'].append(float(time.time() - start_time))
        ## Code to profile ##

for (lang1, lang2) in res_dict_1K: 
    res_dict_1K[(lang1, lang2)]['avg_runtime'] = np.mean(res_dict_1K[(lang1, lang2)]['runtime'])

json_results = {k[0] + "_" + k[1]: v for k, v in res_dict_1K.items()}
with open('runtime_reprod_1K.json', 'w') as fp: 
    json.dump(json_results, fp)