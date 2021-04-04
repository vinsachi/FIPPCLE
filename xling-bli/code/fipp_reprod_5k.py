import os

res_dict_5K = {('de', 'fi'): {'eps': 0.05, 'lamb': 0.25, 'paper_mrr': 0.389041447728},
 ('de', 'fr'): {'eps': 0.025, 'lamb': 1.25, 'paper_mrr': 0.54281909387},
 ('de', 'hr'): {'eps': 0.05, 'lamb': 1.25, 'paper_mrr': 0.359810201265},
 ('de', 'it'): {'eps': 0.1, 'lamb': 1.25, 'paper_mrr': 0.533353901861},
 ('de', 'ru'): {'eps': 0.025, 'lamb': 0.5, 'paper_mrr': 0.449057793892},
 ('de', 'tr'): {'eps': 0.1, 'lamb': 0.25, 'paper_mrr': 0.32130994195},
 ('en', 'de'): {'eps': 0.01, 'lamb': 0.75, 'paper_mrr': 0.589719712832},
 ('en', 'fi'): {'eps': 0.01, 'lamb': 0.25, 'paper_mrr': 0.439063399328},
 ('en', 'fr'): {'eps': 0.025, 'lamb': 1.25, 'paper_mrr': 0.67926603202},
 ('en', 'hr'): {'eps': 0.025, 'lamb': 0.75, 'paper_mrr': 0.381594128371},
 ('en', 'it'): {'eps': 0.05, 'lamb': 0.5, 'paper_mrr': 0.648637337125},
 ('en', 'ru'): {'eps': 0.025, 'lamb': 0.5, 'paper_mrr': 0.502331720094},
 ('en', 'tr'): {'eps': 0.025, 'lamb': 0.25, 'paper_mrr': 0.406944659103},
 ('fi', 'fr'): {'eps': 0.1, 'lamb': 1.25, 'paper_mrr': 0.407488745618},
 ('fi', 'hr'): {'eps': 0.1, 'lamb': 1.25, 'paper_mrr': 0.334558293872},
 ('fi', 'it'): {'eps': 0.1, 'lamb': 0.5, 'paper_mrr': 0.407218919951},
 ('fi', 'ru'): {'eps': 0.05, 'lamb': 1.0, 'paper_mrr': 0.378641297901},
 ('hr', 'fr'): {'eps': 0.05, 'lamb': 1.25, 'paper_mrr': 0.426307035126},
 ('hr', 'it'): {'eps': 0.01, 'lamb': 0.25, 'paper_mrr': 0.415387115087},
 ('hr', 'ru'): {'eps': 0.01, 'lamb': 0.75, 'paper_mrr': 0.408165611616},
 ('it', 'fr'): {'eps': 0.05, 'lamb': 1.25, 'paper_mrr': 0.683967894951},
 ('ru', 'fr'): {'eps': 0.05, 'lamb': 0.25, 'paper_mrr': 0.497144990129},
 ('ru', 'it'): {'eps': 0.025, 'lamb': 1.0, 'paper_mrr': 0.502551096883},
 ('tr', 'fi'): {'eps': 0.15, 'lamb': 0.75, 'paper_mrr': 0.306043368909},
 ('tr', 'fr'): {'eps': 0.025, 'lamb': 1.25, 'paper_mrr': 0.379930038423},
 ('tr', 'hr'): {'eps': 0.1, 'lamb': 0.75, 'paper_mrr': 0.28815146609},
 ('tr', 'it'): {'eps': 0.01, 'lamb': 0.75, 'paper_mrr': 0.371963387864},
 ('tr', 'ru'): {'eps': 0.1, 'lamb': 0.5, 'paper_mrr': 0.318536782223}}

for (lang1, lang2) in res_dict_5K:
	eps, lamb = res_dict_5K[lang1, lang2]['eps'], res_dict_5K[lang1, lang2]['lamb']
	os.system('python map.py -m p -d bli_datasets/%s-%s/yacle.train.freq.5k.%s-%s.tsv --lang_src %s --lang_trg %s ft-raw-200k/vecs_%s ft-raw-200k/vocab_%s ft-raw-200k/vecs_%s ft-raw-200k/vocab_%s ft-raw-200k/ --eps %s --lamb %s'% (lang1, lang2, lang1, lang2, lang1, lang2, lang1, lang1, lang2, lang2, eps, lamb))
	os.system('python eval.py bli_datasets/%s-%s/yacle.test.freq.2k.%s-%s.tsv ft-raw-200k/%s-%s.%s.vectors ft-raw-200k/%s-%s.%s.vectors ft-raw-200k/%s-%s.%s.vocab ft-raw-200k/%s-%s.%s.vocab --eps %s --lamb %s'%(lang1, lang2, lang1, lang2, lang1, lang2, lang1, lang1, lang2, lang2, lang1, lang2, lang1, lang1, lang2, lang2, eps, lamb))

	f = open("results_tuned/%s-%s_results_%s_%s.txt"%(lang1, lang2, eps, lamb), "r")
	res_dict_5K[(lang1, lang2)]['result_mrr'] = float(f.readlines()[-1])

print(res_dict_5K)