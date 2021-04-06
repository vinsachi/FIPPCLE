import os
import json

res_dict_5K = {('de', 'fi'): {'eps': 0.05,   'lamb': 0.25,  'paper_mrr': 0.389},
               ('de', 'fr'): {'eps': 0.025,  'lamb': 1.25,  'paper_mrr': 0.543},
               ('de', 'hr'): {'eps': 0.05,   'lamb': 1.25,  'paper_mrr': 0.360},
               ('de', 'it'): {'eps': 0.1,    'lamb': 1.25,  'paper_mrr': 0.533},
               ('de', 'ru'): {'eps': 0.025,  'lamb': 0.5,   'paper_mrr': 0.449},
               ('de', 'tr'): {'eps': 0.1,    'lamb': 0.25,  'paper_mrr': 0.321},
               ('en', 'de'): {'eps': 0.01,   'lamb': 0.75,  'paper_mrr': 0.590},
               ('en', 'fi'): {'eps': 0.01,   'lamb': 0.25,  'paper_mrr': 0.439},
               ('en', 'fr'): {'eps': 0.025,  'lamb': 1.25,  'paper_mrr': 0.679},
               ('en', 'hr'): {'eps': 0.025,  'lamb': 0.75,  'paper_mrr': 0.382},
               ('en', 'it'): {'eps': 0.05,   'lamb': 0.5,   'paper_mrr': 0.649},
               ('en', 'ru'): {'eps': 0.025,  'lamb': 0.5,   'paper_mrr': 0.502},
               ('en', 'tr'): {'eps': 0.025,  'lamb': 0.25,  'paper_mrr': 0.407},
               ('fi', 'fr'): {'eps': 0.1,    'lamb': 1.25,  'paper_mrr': 0.407},
               ('fi', 'hr'): {'eps': 0.1,    'lamb': 1.25,  'paper_mrr': 0.335},
               ('fi', 'it'): {'eps': 0.1,    'lamb': 0.5,   'paper_mrr': 0.407},
               ('fi', 'ru'): {'eps': 0.05,   'lamb': 1.0,   'paper_mrr': 0.379},
               ('hr', 'fr'): {'eps': 0.05,   'lamb': 1.25,  'paper_mrr': 0.426},
               ('hr', 'it'): {'eps': 0.01,   'lamb': 0.25,  'paper_mrr': 0.415},
               ('hr', 'ru'): {'eps': 0.01,   'lamb': 0.75,  'paper_mrr': 0.408},
               ('it', 'fr'): {'eps': 0.05,   'lamb': 1.25,  'paper_mrr': 0.684},
               ('ru', 'fr'): {'eps': 0.05,   'lamb': 0.25,  'paper_mrr': 0.497},
               ('ru', 'it'): {'eps': 0.025,  'lamb': 1.0,   'paper_mrr': 0.503},
               ('tr', 'fi'): {'eps': 0.15,   'lamb': 0.75,  'paper_mrr': 0.306},
               ('tr', 'fr'): {'eps': 0.025,  'lamb': 1.25,  'paper_mrr': 0.380},
               ('tr', 'hr'): {'eps': 0.1,    'lamb': 0.75,  'paper_mrr': 0.288},
               ('tr', 'it'): {'eps': 0.01,   'lamb': 0.75,  'paper_mrr': 0.372},
               ('tr', 'ru'): {'eps': 0.1,    'lamb': 0.5,   'paper_mrr': 0.319}}

for (lang1, lang2) in res_dict_5K:
	eps, lamb = res_dict_5K[lang1, lang2]['eps'], res_dict_5K[lang1, lang2]['lamb']
	os.system('python code/map.py -m f -d bli_datasets/%s-%s/yacle.train.freq.5k.%s-%s.tsv --lang_src %s --lang_trg %s ft-raw-200k/vecs_%s ft-raw-200k/vocab_%s ft-raw-200k/vecs_%s ft-raw-200k/vocab_%s ft-raw-200k/ --eps %s --lamb %s'% (lang1, lang2, lang1, lang2, lang1, lang2, lang1, lang1, lang2, lang2, eps, lamb))
	os.system('python code/eval.py bli_datasets/%s-%s/yacle.test.freq.2k.%s-%s.tsv ft-raw-200k/%s-%s.%s.vectors ft-raw-200k/%s-%s.%s.vectors ft-raw-200k/%s-%s.%s.vocab ft-raw-200k/%s-%s.%s.vocab --eps %s --lamb %s'%(lang1, lang2, lang1, lang2, lang1, lang2, lang1, lang1, lang2, lang2, lang1, lang2, lang1, lang1, lang2, lang2, eps, lamb))

	f = open("results_tuned/%s-%s_results_%s_%s.txt"%(lang1, lang2, eps, lamb), "r")
	res_dict_5K[(lang1, lang2)]['result_mrr'] = float(f.readlines()[-1])

json_results = {k[0] + "_" + k[1] : v for k, v in res_dict_5K.items()}
with open('results_5K.json', 'w') as fp: 
    json.dump(json_results, fp)
