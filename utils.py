import os
import sys
import re
import numpy as np
import random
import csv
import pickle
random_seed = 7
np.random.seed(random_seed)
random.seed(random_seed)
import itertools
from collections import Counter

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

from pca import PCA
from wpca import WPCA
from numpy.linalg import svd, norm

def writeDebuggingLogs(sentence, debugInfo, ferr):
    if len(debugInfo)==3:
        ferr.write("[No Representation Possible for Context]: None of the sentence words were found in the pretrained embeddings: "+" ".join(sentence)+" "+"_".join(debugInfo)+"\n")
    if len(debugInfo)==2:
        ferr.write("[No Representation Possible for Entity]: None of the description words were found in the pretrained embeddings: "+" ".join(sentence)+" "+"_".join(debugInfo)+"\n")
    if len(debugInfo)==1:
        ferr.write("[No Coherence Representation Possible]: for the entities: "+" ".join(sentence)+" in the document: "+"_".join(debugInfo)+"\n")

def loadWordVectors(path):
    if "bin" in path:
        isBinary = True
    else:
        isBinary = False
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=isBinary)
    return word_vectors

def loadWikipedia2VecVectors(path):
    wikipedia2vec_vectors = pickle.load(open(path,"rb"))
    return wikipedia2vec_vectors

def loadGaneaVectors(path, redirect=False, glove=False):
    ganea_vectors = {}
    if glove==True:
        wv = np.load(path+'glove/word_embeddings.npy')
        words = open(path+'glove/dict.word')
    else:
        wv = np.load(path+'word_embeddings.npy')
        words = open(path+'dict.word')
    ev = np.load(path+'entity_embeddings.npy')
    if redirect==True:
        entities = open(path+'dict_redirects.entity')
    else:
        entities = open(path+'dict.entity')
    it = 0
    for line in words:
        line = line.strip().split("\t")
        word = line[0].strip()
        ganea_vectors[word] = wv[it]
        it+=1
    it = 0
    for line in entities:
        line = line.strip().split("\t")
        entity = line[0].strip().split("wiki/")[-1]
        ganea_vectors["ENTITY/"+entity] = ev[it]
        it+=1
    return ganea_vectors

def cosineSimilarity(vector1, vector2):
    if (norm(vector1) * norm(vector2)) == 0:
        return 0

    return np.dot(vector1, vector2) / (norm(vector1) * norm(vector2))

def computeVecSubspaceSimilarity(vector, subspace, singularValues, weighted=False):
    if subspace.ndim == 1:
        return cosineSimilarity(vector, subspace)
    if weighted == True:
        mat = np.dot(vector, np.multiply(subspace.T,singularValues))/np.sum(singularValues) # scaling each singular vector by the corresponding singular values, thus, valuing their contribution more towards the overall similarity
    else:
        mat = np.dot(vector, subspace.T)
    if norm(vector) != 0.0:
        sim = (np.sum(mat ** 2) ** 0.5)/norm(vector)
    else:
        sim = 0.0
    return sim

def computeSubspaceSimilarity(subspace1, subspace2):
    if subspace1.ndim == 1 and subspace2.ndim != 1:
        # subspace1 is the vector
        sim = computeVecSubspaceSimilarity(subspace1,subspace2,None)
    elif subspace1.ndim != 1 and subspace2.ndim == 1:
        # subspace2 is the vector
        sim = computeVecSubspaceSimilarity(subspace2,subspace1,None)
    elif subspace1.ndim == 1 and subspace2.ndim == 1:
        sim = (cosineSimilarity(subspace1,subspace2)**2)**0.5 # sqrt of (sum of squares of cosSims)
    else: # Figure out a way to normalize the similairity score (between 0 and 1)
        # compute principal angles between the subspaces
        mat = np.dot(subspace1, subspace2.T)
        # mean centre the data??
        #mat -= np.mean(mat,axis=0)
        singularValues = svd(mat,compute_uv=False)
        sim = np.sqrt(np.sum(singularValues ** 2,axis=0))
    return sim

def constructEmbeddingMatrix(elements, weights, vectors):
    embedding_matrix = np.array([]) # Initialize with an empty size 0 array
    weight_array = []

    if vectors.__class__.__name__ == 'Word2VecKeyedVectors': # pre-trained word-vectors (w2v, glove, fasttext)
        dim = vectors['the'].shape[0] # [hack] Better to fix the number of dimensions of the embeddings from the code
    else: # custom embedding dictionary
        dim = list(vectors.values())[0].shape[0]

    element_weight_pairs = zip(elements, weights)
    for (element, weight) in element_weight_pairs:
        try:
            if embedding_matrix.size == 0:
                embedding_matrix = vectors[element]/norm(vectors[element])
            else:
                embedding_matrix = np.vstack((embedding_matrix,vectors[element]/norm(vectors[element])))
            weight_array.append(weight) # keep weights only for those elements existent in the supplied embedding vectors
        except KeyError:
            continue
    weight_array = np.array(weight_array).astype(np.float)
    return embedding_matrix, weight_array, dim

def average(elements, embedding_matrix, weight_array, vector_dim, debugInfo):
    ferr = open("errors_average_representation","a+")
    embedding_matrix = (embedding_matrix.T * weight_array).T
    if embedding_matrix.size == 0: # assign a vector of all 0s to the representation in this case
        representation = np.zeros((vector_dim,))
        if debugInfo: # Write Debugging Logs
            writeDebuggingLogs(elements, debugInfo, ferr)
    elif embedding_matrix.ndim == 1:
        representation = embedding_matrix
    else:
        #representation = np.mean(embedding_matrix,axis=0)
        representation = np.sum(embedding_matrix,axis=0)/np.sum(weight_array)
    ferr.close()
    return representation

def pca_subspace(elements, embedding_matrix, vector_dim, mean_centering, numComponents, debugInfo):
    ferr = open("errors_pca_representation","a+")
    flog = open("logs_pca_representation","a+")

    if embedding_matrix.ndim == 1: # only one word in the sentence, do nothing (no PCA), the vector-space of the word itself is the subspace
        ferr.write("[No PCA]: Only a single element from "+" ".join(elements)+" found in supplied embeddings for the document"+"_".join(debugInfo)+"\n")
        subspace = embedding_matrix; singularValues = np.array([1.0]); energyRetained = 1.0
    else:
        flog.write("Original NumComponents: "+str(numComponents)+" NumElements: "+str(embedding_matrix.shape[0])+"\t")
        numComponents = min(embedding_matrix.shape[0], embedding_matrix.shape[1], numComponents)
        flog.write("New NumComponents: "+str(numComponents)+"\n")

        pca = PCA(n_components=numComponents, mean_centering=mean_centering)
        try:
            pca.fit(embedding_matrix)
            subspace = pca.components_
            if numComponents == 1: # convert matrix to vector when numComponents = 1
                subspace = subspace.T.reshape(-1)
            energyRetained = np.sum(pca.explained_variance_ratio_)
            singularValues = pca.singular_values_
        except (np.linalg.LinAlgError, ZeroDivisionError) as e: # Fails (svd doesn't converge) for some reason. Use the word-vector average in this case!
            ferr.write("[SVD Error]: No subspace constructed for "+" ".join(elements)+" in the document: "+"_".join(debugInfo)+"\n")
            subspace = np.mean(embedding_matrix,axis=0); singularValues = np.array([1.0]); energyRetained = 1.0
    ferr.close()
    flog.close()
    return subspace, singularValues, energyRetained

def wpca_subspace(elements, embedding_matrix, weight_array, vector_dim, mean_centering, numComponents, debugInfo):
    ferr = open("errors_wpca_representation","a+")
    flog = open("logs_pca_representation","a+")
    weight_matrix = np.tile(weight_array.reshape(-1,1), vector_dim)

    if embedding_matrix.ndim == 1: # only one word in the sentence, do nothing (no PCA), the vector-space of the word itself is the subspace
        ferr.write("[No WPCA]: Only a single element from "+" ".join(elements)+" found in supplied embeddings for the document"+"_".join(debugInfo)+"\n")
        subspace = embedding_matrix; singularValues = np.array([1.0]); energyRetained = 1.0
    else:
        flog.write("Original NumComponents: "+str(numComponents)+" NumElements: "+str(embedding_matrix.shape[0])+"\t")
        numComponents = min(embedding_matrix.shape[0], embedding_matrix.shape[1], numComponents)
        flog.write("New NumComponents: "+str(numComponents)+"\n")

        pca = WPCA(n_components=numComponents, mean_centering=mean_centering) #WPCA centers the matrix automatically
        try:
            kwds = {'weights': weight_matrix}
            pca.fit(embedding_matrix, **kwds)
            subspace = pca.components_
            if numComponents == 1: # convert matrix to vector when numComponents = 1
                subspace = subspace.T.reshape(-1)
            energyRetained = np.sum(pca.explained_variance_ratio_)

            if np.any(pca.explained_variance_ < 0):  # Hack
                explained_variance = np.abs(pca.explained_variance_)
                ferr.write("[Numerical Precision Error]: Negative variance "+str(pca.explained_variance_)+" in subspace constructed for "+" ".join(elements)+" in the document: "+"_".join(debugInfo)+"\n")
            else:
                explained_variance = pca.explained_variance_
            #singularValues = np.sqrt( explained_variance * (embedding_matrix.shape[0] - 1) )
            singularValues = np.sqrt( explained_variance )
        except (np.linalg.LinAlgError, ZeroDivisionError) as e: # Fails (svd doesn't converge) for some reason. Use the word-vector average in this case!
            ferr.write("[WPCA Error]: No subspace constructed for "+" ".join(elements)+" in the document: "+"_".join(debugInfo)+"\n")
            subspace = np.mean(embedding_matrix,axis=0); singularValues = np.array([1.0]); energyRetained = 1.0
    ferr.close()
    flog.close()
    return subspace, singularValues, energyRetained

def constructRepresentation(elements, weights, vectors, mode, mean_centering=True, numComponents=None, debugInfo=None):
    embedding_matrix, weight_array, vector_dim = constructEmbeddingMatrix(elements, weights, vectors)

    if mode == 'pca':
        return pca_subspace(elements, embedding_matrix, vector_dim, mean_centering, numComponents, debugInfo)
    elif mode == 'wpca':
        return wpca_subspace(elements, embedding_matrix, weight_array, vector_dim, mean_centering, numComponents, debugInfo)
    else:
        return average(elements, embedding_matrix, weight_array, vector_dim, debugInfo)
