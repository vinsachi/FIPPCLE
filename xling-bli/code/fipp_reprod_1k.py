import os

res_dict_1K = {('de', 'fi'): {'eps': 0.05, 'lamb': 1.25, 'paper_mrr': 0.296256270839},
('de', 'fr'): {'eps': 0.05, 'lamb': 1.25, 'paper_mrr': 0.46288858164},
('de', 'hr'): {'eps': 0.1, 'lamb': 1.0, 'paper_mrr': 0.268177633164},
('de', 'it'): {'eps': 0.01, 'lamb': 0.25, 'paper_mrr': 0.481618050717},
('de', 'ru'): {'eps': 0.1, 'lamb': 1.0, 'paper_mrr': 0.359251122569},
('de', 'tr'): {'eps': 0.1, 'lamb': 0.75, 'paper_mrr': 0.21488245162},
('en', 'de'): {'eps': 0.05, 'lamb': 0.75, 'paper_mrr': 0.513424530572},
('en', 'fi'): {'eps': 0.025, 'lamb': 1.25, 'paper_mrr': 0.314084927353},
('en', 'fr'): {'eps': 0.1, 'lamb': 0.5, 'paper_mrr': 0.600669758026},
('en', 'hr'): {'eps': 0.05, 'lamb': 1.25, 'paper_mrr': 0.274760579659},
('en', 'it'): {'eps': 0.05, 'lamb': 1.0, 'paper_mrr': 0.591269915471},
('en', 'ru'): {'eps': 0.05, 'lamb': 1.25, 'paper_mrr': 0.399190629658},
('en', 'tr'): {'eps': 0.1, 'lamb': 0.5, 'paper_mrr': 0.291999120576},
('fi', 'fr'): {'eps': 0.1, 'lamb': 1.25, 'paper_mrr': 0.274045419819}, 
('fi', 'hr'): {'eps': 0.05, 'lamb': 1.25, 'paper_mrr': 0.24337024039},
('fi', 'it'): {'eps': 0.15, 'lamb': 0.75, 'paper_mrr': 0.309320229772},
('fi', 'ru'): {'eps': 0.1, 'lamb': 1.0, 'paper_mrr': 0.28533941901},
('hr', 'fr'): {'eps': 0.1, 'lamb': 0.75, 'paper_mrr': 0.283437622014},
('hr', 'it'): {'eps': 0.1, 'lamb': 1.25, 'paper_mrr': 0.317915989553},
('hr', 'ru'): {'eps': 0.1, 'lamb': 0.5, 'paper_mrr': 0.318228888511},
('it', 'fr'): {'eps': 0.025, 'lamb': 1.25, 'paper_mrr': 0.63850292885},
('ru', 'fr'): {'eps': 0.05, 'lamb': 1.0, 'paper_mrr': 0.382728908926},
('ru', 'it'): {'eps': 0.1, 'lamb': 0.75, 'paper_mrr': 0.413300568695},
('tr', 'fi'): {'eps': 0.15, 'lamb': 1.25, 'paper_mrr': 0.199961030488},
('tr', 'fr'): {'eps': 0.1, 'lamb': 1.25, 'paper_mrr': 0.251119590303},
('tr', 'hr'): {'eps': 0.15, 'lamb': 0.5, 'paper_mrr': 0.183575671939},
('tr', 'it'): {'eps': 0.1, 'lamb': 0.75, 'paper_mrr': 0.263392418121},
('tr', 'ru'): {'eps': 0.1, 'lamb': 1.25, 'paper_mrr': 0.205270061343}}

for (lang1, lang2) in res_dict_1K:
	eps, lamb = res_dict_1K[lang1, lang2]['eps'], res_dict_1K[lang1, lang2]['lamb']
	os.system('python map.py -m f -d bli_datasets/%s-%s/yacle.train.freq.1k.%s-%s.tsv --lang_src %s --lang_trg %s ft-raw-200k/vecs_%s ft-raw-200k/vocab_%s ft-raw-200k/vecs_%s ft-raw-200k/vocab_%s ft-raw-200k/ --eps %s --lamb %s --self_learn_num 14000'% (lang1, lang2, lang1, lang2, lang1, lang2, lang1, lang1, lang2, lang2, eps, lamb))
	os.system('python eval.py bli_datasets/%s-%s/yacle.test.freq.2k.%s-%s.tsv ft-raw-200k/%s-%s.%s.vectors ft-raw-200k/%s-%s.%s.vectors ft-raw-200k/%s-%s.%s.vocab ft-raw-200k/%s-%s.%s.vocab --eps %s --lamb %s'%(lang1, lang2, lang1, lang2, lang1, lang2, lang1, lang1, lang2, lang2, lang1, lang2, lang1, lang1, lang2, lang2, eps, lamb))

	f = open("results_tuned/%s-%s_results_%s_%s.txt"%(lang1, lang2, eps, lamb), "r")
	res_dict_1K[(lang1, lang2)]['result_mrr'] = float(f.readlines()[-1])

print(res_dict_1K)