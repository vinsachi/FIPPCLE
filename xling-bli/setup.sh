# (I) create results directory
mkdir results_tuned

# (II) download fastText embeddings (courtesy of Prof. Glavas)
# https://drive.google.com/drive/folders/1RUT8qEcU4drlEHDcxPAbzUNwcUP3SYyg
pip install gdown
mkdir ft-raw-200k
cd ft-raw-200k

# Each of the 8 downloads corresponds to the languages in the XLING-BLI dataset
gdown https://drive.google.com/uc?id=1YeQFauynj3JRwV6svWdyy0fMBwYMKNNd # de
gdown https://drive.google.com/uc?id=1wy-OlRycb5_dEB0w52dkOSJqfEXDytDT # en
gdown https://drive.google.com/uc?id=1wHJJk8yKu0yWsCi7wRf0SoWb_YwyZbBi # fi
gdown https://drive.google.com/uc?id=1nK5kdpUvG8L3q9q2IqTFFZnqdzKCgHGo # fr
gdown https://drive.google.com/uc?id=1fz1R3vayiIHd3OIfDS3HJ1yChCOrQguj # hr
gdown https://drive.google.com/uc?id=1x6tL-Hh7LVQ0KbMiWe9DI4TO1uRBFghM # it
gdown https://drive.google.com/uc?id=12pIp9zsLmeF5O-512Q1OmohU3AMVlU7a # ru
gdown https://drive.google.com/uc?id=10rryBWx5KWA136utnrEF99MJNB4h2-Xj # tr


# (III) Load and serialize embeddings (follows setup in codogogo/xling-eval)
cd ../code
python emb_serializer.py ../ft-raw-200k/fasttext.wiki.de.300.vocab_200K.vec ../ft-raw-200k/vocab_de ../ft-raw-200k/vecs_de
python emb_serializer.py ../ft-raw-200k/fasttext.wiki.en.300.vocab_200K.vec ../ft-raw-200k/vocab_en ../ft-raw-200k/vecs_en
python emb_serializer.py ../ft-raw-200k/fasttext.wiki.fi.300.vocab_200K.vec ../ft-raw-200k/vocab_fi ../ft-raw-200k/vecs_fi
python emb_serializer.py ../ft-raw-200k/fasttext.wiki.fr.300.vocab_200K.vec ../ft-raw-200k/vocab_fr ../ft-raw-200k/vecs_fr
python emb_serializer.py ../ft-raw-200k/fasttext.wiki.hr.300.vocab_200K.vec ../ft-raw-200k/vocab_hr ../ft-raw-200k/vecs_hr
python emb_serializer.py ../ft-raw-200k/fasttext.wiki.it.300.vocab_200K.vec ../ft-raw-200k/vocab_it ../ft-raw-200k/vecs_it
python emb_serializer.py ../ft-raw-200k/fasttext.wiki.ru.300.vocab_200K.vec ../ft-raw-200k/vocab_ru ../ft-raw-200k/vecs_ru
python emb_serializer.py ../ft-raw-200k/fasttext.wiki.tr.300.vocab_200K.vec ../ft-raw-200k/vocab_tr ../ft-raw-200k/vecs_tr

# (IV) Install python dependencies 
# when using self-learning, please install torch from https://pytorch.org based on your hardware specifications
pip install numpy scipy tqdm IPython scikit-learn

# (V) Run reproduction scripts and output to json
cd ..
python fipp_reprod_1k.py
python fipp_reprod_5k.py