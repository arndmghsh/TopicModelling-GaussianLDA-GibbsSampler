import pickle
import numpy as np
from numpy import pi, log
from scipy.special import gammaln

def load_embeddings():
    """Loads vector representations for vocabulary"""
    print("Loading embeddings...")
    embeddings = np.load("news_embeddings.npy")
    print(embeddings.shape)
    with open('news_word2index.pkl', 'rb') as f:
        word2index = pickle.load(f)
    return embeddings, word2index

def load_corpus():
    """Loads matrix with each row as a BoW representation of a document"""
    print("Loading corpus...")
    corpus = np.load("news_corpus.npy")
    return corpus

def multivariate_t_distribution(x, mu, nu, M):
    '''
    Multivariate t-student density with fixed Sigma = I:
    output:
        the density of the given element
    input:
        x = parameter (d dimensional numpy array or scalar)
        mu = mean (d dimensional numpy array or scalar)
        nu = degrees of freedom
        M: dimension
    '''
    inv_sigma = np.eye(50)
    det = 1
    matrix = (((x-mu)[None])@inv_sigma@((x-mu)[None].T))[0][0]
    log_prob = gammaln((nu + M) / 2) \
    - (gammaln(nu / 2) + M/2*(log(nu) + log(pi)) + 0.5 * log(det) + ((nu + M)/2) * log(1 + matrix / nu))
    return log_prob