import numpy as np
import cca
import util
from operator import itemgetter
from IPython import embed
import scipy
import operator
import tqdm
import torch

import numpy as np
from sklearn.decomposition import PCA

def isotrop_preproc(v, D = 1):
      """
      Code from https://gist.github.com/lgalke/febaaa1313d9c11f3bc8240defed8390
      Arguments:
          :v: word vectors of shape (n_words, n_dimensions)
          :D: number of principal components to subtract
      """
      # 1. Subtract mean vector
      v_tilde = v - np.mean(v, axis=0)
      # 2. Compute the first `D` principal components
      #    on centered embedding vectors
      u = PCA(n_components=D).fit(v_tilde).components_  # [D, emb_size]
      # Subtract first `D` principal components
      # [vocab_size, emb_size] @ [emb_size, D] @ [D, emb_size] -> [vocab_size, emb_size]
      return v_tilde - (np.matmul(v, np.matmul(u.T, u)))

def get_seeds(vocab_dict_src, vocab_dict_trg, n = 5000):
  allmatches = [k for k in vocab_dict_src if k in vocab_dict_trg]
  allmatches.sort(key = lambda x: vocab_dict_src[x] + vocab_dict_trg[x])
  return allmatches[:n]

def build_matrices(vocab_dict_src, vocab_dict_trg, embs_src, embs_trg, trans_dict = None, num_same = 5000):
  src_mat = []
  trg_mat = []
  if trans_dict:
    for sw, tw in trans_dict:
      if sw in vocab_dict_src and tw in vocab_dict_trg:
        src_mat.append(embs_src[vocab_dict_src[sw]])
        trg_mat.append(embs_trg[vocab_dict_trg[tw]])
  else:
    seeds = get_seeds(vocab_dict_src, vocab_dict_trg, n = num_same)
    for s in seeds:
      src_mat.append(embs_src[vocab_dict_src[s]])
      trg_mat.append(embs_trg[vocab_dict_trg[s]])
  return np.array(src_mat, dtype=np.float32), np.array(trg_mat, dtype=np.float32)

def project_pinv(vocab_dict_src, embs_src, vocab_dict_trg, embs_trg, trans_dict = None):
  src_mat, trg_mat = build_matrices(vocab_dict_src, vocab_dict_trg, embs_src, embs_trg, trans_dict)
  proj_mat = np.dot(np.linalg.pinv(src_mat), trg_mat)
  return np.dot(embs_src, proj_mat), proj_mat

def project_cca(vocab_dict_src, embs_src, vocab_dict_trg, embs_trg, trans_dict = None):
  src_mat, trg_mat = build_matrices(vocab_dict_src, vocab_dict_trg, embs_src, embs_trg, trans_dict)
  corr_an = cca.CCA(src_mat, trg_mat, min(src_mat.shape[1], trg_mat.shape[1]))
  corr_an.correlate(sklearn = False)
  proj_src, proj_trg = corr_an.transform(embs_src, embs_trg)
  return proj_src, proj_trg, corr_an

def project_proc(vocab_dict_src, embs_src, vocab_dict_trg, embs_trg, trans_dict = None):
  src_mat, trg_mat = build_matrices(vocab_dict_src, vocab_dict_trg, embs_src, embs_trg, trans_dict)
  product = np.matmul(src_mat.transpose(), trg_mat)
  U, s, V = np.linalg.svd(product)
  proj_mat = np.matmul(U, V)

  embs_src_projected = np.matmul(embs_src, proj_mat)  
  return embs_src_projected, proj_mat, src_mat.shape[0]

def project_fipp(vocab_dict_src, embs_src, vocab_dict_trg, embs_trg, trans_dict, eps = 0.05, lamb = 1.0, self_learn_num = 14000, sl_chunk_size = 100):
  ## (1) Preprocess embeddings ##
  embs_src /= np.linalg.norm(embs_src, axis=1)[:, np.newaxis]
  embs_trg /= np.linalg.norm(embs_trg, axis=1)[:, np.newaxis]
  embs_src = isotrop_preproc(embs_src)
  embs_trg = isotrop_preproc(embs_trg)
  src_train, tgt_train = build_matrices(vocab_dict_src, vocab_dict_trg, embs_src, embs_trg, trans_dict)
  ## (1) Preprocess embeddings ##

  ## (2) Self-learning framework ##
  if self_learn_num > 0:
    # Get indices of source and target pairs from training set
    curr_idxs_src, curr_idxs_tgt, max_sims = [], [], []
    for sw, tw in trans_dict:
        if sw in vocab_dict_src and tw in vocab_dict_trg:
            curr_idxs_src.append(vocab_dict_src[sw])
            curr_idxs_tgt.append(vocab_dict_trg[tw])

    # Get normalized similarity matrices for training pairs
    sims_mat_src, sims_mat_tgt = np.dot(src_train, embs_src.T).T, np.dot(tgt_train, embs_trg.T).T
    sims_mat_src /= np.linalg.norm(sims_mat_src, axis=1)[:, np.newaxis]
    sims_mat_tgt /= np.linalg.norm(sims_mat_tgt, axis=1)[:, np.newaxis]

    # Get cross similarity between source and target words
    device = torch.device("cuda")
    trg_mat = torch.Tensor(sims_mat_tgt.T).to(device)
    for chunk in tqdm.tqdm(range(int(len(sims_mat_src)/sl_chunk_size)+1)):
        sim_vec = torch.matmul(torch.Tensor(sims_mat_src[sl_chunk_size*chunk:sl_chunk_size*(chunk+1)]).to(device), trg_mat)
        
        batch_sims = torch.max(sim_vec, dim = 1)
        for idx in range(len(batch_sims[0])): 
            max_sims.append((idx + chunk * sl_chunk_size, float(batch_sims[0][idx]), int(batch_sims[1][idx])))

    # Augment training set using pairs with most similar pairs outside of training set
    max_sims.sort(key=lambda x:x[1], reverse = True)
    total_augs, idx_to_word_src, idx_to_word_tgt = 0, {v:k for k,v in vocab_dict_src.items()}, {v:k for k,v in vocab_dict_trg.items()}

    for (idx_src, sim, idx_tgt) in max_sims:
        if idx_src not in curr_idxs_src and idx_tgt not in curr_idxs_tgt:
            total_augs += 1
            curr_idxs_src.append(idx_src)
            curr_idxs_tgt.append(idx_tgt)
            src_train = np.vstack([src_train, embs_src[idx_src]])
            tgt_train = np.vstack([tgt_train, embs_trg[idx_tgt]])

        if total_augs > self_learn_num:
            break
  ## (2) Self-learning framework ##

  ## (3) Run FIPP ##
  train_samples = len(src_train)
  x_s_inner_prod, x_t_inner_prod = np.matmul(src_train, src_train.T), np.matmul(tgt_train, tgt_train.T)
  indicator_mat = np.where(np.greater_equal(np.abs(x_s_inner_prod - x_t_inner_prod), eps),
                          np.zeros((train_samples, train_samples)),
                          np.ones((train_samples, train_samples)))

  gamma = (train_samples**2)/np.sum(indicator_mat)
  gram_fipp = np.where(np.greater(indicator_mat, 0), 
                          (x_s_inner_prod + (gamma * lamb * x_t_inner_prod)) / (1.0 + gamma * lamb), 
                          x_s_inner_prod)

  eig_vals, eig_vecs = scipy.linalg.eigh(gram_fipp, eigvals = (train_samples - embs_trg.shape[1], train_samples - 1))
  x_s_tilde_result = np.matmul(eig_vecs, np.sqrt(np.diag(eig_vals)))

  train_test_sim_mat, dim_mat = np.dot(src_train, embs_src.T), x_s_tilde_result.T.dot(x_s_tilde_result)
  ls_proj_x_s_tilde = scipy.linalg.solve(dim_mat, x_s_tilde_result.T.dot(train_test_sim_mat), assume_a = 'pos')
  ## (3) Run FIPP ##

  ## (4) Weighted procrustes rotation  ##
  weight_vec = (1.0/np.linalg.norm(gram_fipp - x_t_inner_prod, axis = 0))[:, np.newaxis]
  src_train = ls_proj_x_s_tilde.T[curr_idxs_src]

  product = np.matmul((weight_vec * src_train).T, (weight_vec * tgt_train))
  U, s, V = np.linalg.svd(product)
  proj_mat = np.matmul(U, V)
  embs_src_projected = np.matmul(ls_proj_x_s_tilde.T, proj_mat)
  ## (4) Weighted procrustes rotation  ##
  return embs_src_projected, proj_mat, embs_trg


def project_proc_bootstrap(vocab_dict_src, embs_src, vocab_dict_trg, embs_trg, trans_dict = None, growth_rate = 1.5, limit = 10000):
  vocab_dict_src_inv = {v : k for k, v in vocab_dict_src.items()}
  vocab_dict_trg_inv = {v : k for k, v in vocab_dict_trg.items()} 
  cnt = 0

  orig_src_norm = util.mat_normalize(embs_src, norm_order=2, axis=1)
  orig_trg_norm = util.mat_normalize(embs_trg, norm_order=2, axis=1) 

  size = 0
  while True:
    cnt += 1
    print("Boostrap iteration: " + str(cnt))
    
    embs_src_projected, _, size1 = project_proc(vocab_dict_src, embs_src, vocab_dict_trg, embs_trg, trans_dict)
    embs_trg_projected, _, size2 = project_proc(vocab_dict_trg, embs_trg, vocab_dict_src, embs_src, [(x[1], x[0]) for x in trans_dict])
    
    if size1 < 1.01 * size or size1 >= limit:
      break
    else:
      size = size1

    proj_src_norm = util.mat_normalize(embs_src_projected, norm_order=2, axis=1)
    proj_trg_norm = util.mat_normalize(embs_trg_projected, norm_order=2, axis=1)
    
    sims_ind_src_trg = util.big_matrix_multiplication(proj_src_norm, orig_trg_norm.transpose(), lambda x: np.argmax(x, axis = 1), chunk_size = 30000)
    sims_ind_trg_src = util.big_matrix_multiplication(proj_trg_norm, orig_src_norm.transpose(), lambda x: np.argmax(x, axis = 1), chunk_size = 30000)
  
    matches = [i for i in range(len(sims_ind_src_trg)) if sims_ind_trg_src[sims_ind_src_trg[i]] == i]

    rank_pairs = [(m, sims_ind_src_trg[m]) for m in matches]
    rank_pairs.sort(key=lambda x: x[0] + x[1])
    cnt = min(int(growth_rate * len(trans_dict)), limit)
    
    if cnt < len(rank_pairs):
      rank_pairs = rank_pairs[:cnt]

    new_trans_dict = [(vocab_dict_src_inv[m[0]], vocab_dict_trg_inv[m[1]]) for m in rank_pairs]
    print(new_trans_dict)
    print("Dict size for next iteration: " + str(len(new_trans_dict)))
    trans_dict = new_trans_dict

  return embs_src_projected, embs_trg