import json
import sys
import pickle
import utils
import numpy as np
import random
import os.path
import pdb
import time
import warnings
warnings.filterwarnings("ignore")

random.seed(110)

STOPWORDS = {'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all',
             'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among',
             'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone',
             'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be',
             'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand',
             'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both', 'bottom',
             'but', 'by', 'call', 'can', 'cannot', 'cant', 'dont', 'co', 'con', 'could', 'couldnt',
             'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg',
             'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even',
             'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen',
             'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty',
             'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had',
             'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein',
             'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred',
             'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself',
             'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may',
             'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly',
             'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless',
             'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now',
             'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other',
             'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per',
             'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming',
             'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six',
             'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes',
             'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their',
             'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore',
             'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though',
             'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward',
             'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very',
             'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever',
             'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether',
             'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will',
             'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves',
             'st', 'years', 'yourselves', 'new', 'used', 'known', 'year', 'later', 'including', 'used',
             'end', 'did', 'just', 'best', 'using'}

def is_important(s, mention_tokens, ctxt_type):
	"""
	an important word is not a stopword, a number, or len == 1
	"""
	word = s[0]; tfidf = s[1]
	if ctxt_type == 'global':
		try:
			if len(word) <= 1 or word.lower() in mention_tokens or tfidf == '0':
				return False
			float(word)
			return False
		except:
			return True
	else:
		try:
			if len(word) <= 1 or word.lower() in STOPWORDS or tfidf == 'None':
				return False
			float(word)
			return False
		except:
			return True

def getContext(left_ctxt, right_ctxt, mention, ctxt_type, ws, doc_name, position, ferr):
	max_len = len(left_ctxt) + len(right_ctxt)
	# choose only window_size number of words from both left and right contextual cues
	if ws != -1:
		context_info = left_ctxt[-ws:] + right_ctxt[:ws]
	else:
		context_info = left_ctxt + right_ctxt

	# removing single letter words, numbers, and stopwords from context
	context_info = [context_pair for context_pair in context_info if is_important(context_pair, mention.lower().split(), ctxt_type)]
	while(len(context_info) == 0):
		if ws == -1 or ws >= max_len:
			context_words = []; tfidf_scores = []
			ferr.write("[Missing Context]: All context words are unimportant for "+doc_name+" "+mention+" "+str(position)+"\n")
			break
		ws = ws * 2
		context_info = left_ctxt[-ws:] + right_ctxt[:ws]
		context_info = [context_pair for context_pair in context_info if is_important(context_pair, mention.lower().split(), ctxt_type)]
	else:
		context_words = list(zip(*context_info))[0]; tfidf_scores = list(zip(*context_info))[1]
	context_words = list(map(lambda x:x.lower(),context_words))
	# As of now all words have equal weight (use the scores from the json file once updated)
	tfidf_scores = [1.0]*len(context_words)
	return context_words, tfidf_scores

def getEntityEmbedding(cand, vectors, dim, doc_name):
	try:
		entity_vector = vectors[cand]
	except KeyError:
		entity_vector = np.zeros((dim,))
	return entity_vector

def computeScores(data, tag, vectors, ferr, params):
	queryId2Mention = {}; mention2QueryId = {}; qid = 1
	isWeighted = params['weight']; isMeanCentered = params['meanCenter']; embeddingType = params['embeddingType']
	num_cands = params['numCands']; num_components = params['ncomp']
	tpca = 0.0; twpca = 0.0
	trueCandMentions = False

	key = []; cand_names = []; hard2beat_baseline = []; avg_baseline = []; wavg_baseline = []; agw_pca = []; agw_wpca = []; labels = []

	for doc_name in data:
		doc_candidates = []; doc_weight_array = []
		doc_entity_candidates = []
		for mention_dict in data[doc_name]:
			mention_name = mention_dict["mention"]
			if 'tabel' in tag:
				position = str(mention_dict["row"])+str(mention_dict["col"])
			else:
				position = mention_dict["posI"]
			true_entity_id = mention_dict["wikidata_id"]
			isDifficult = mention_dict["difficulty"]
			if str(true_entity_id) == '-1':
				ferr.write("[Wikipedia Page for True-Entity has no Wikidata Mapping]: Skip this mention: "+doc_name+" "+mention_name+"\n")
				continue

			if "candidates" in mention_dict:
				candidate_tuples = []
				temp_candidates = []; weight_array = []
				flag = -1; cand_pos = 1

				for cand in mention_dict["candidates"]:
					cand_name = cand[0]
					prominence_score = 1/float(cand_pos)
					try:
						entity_vector = vectors[cand_name]
					except KeyError:
						ferr.write("[Missing Embedding] Skipping candidates that do not have pre-trained entity embeddings: "+doc_name+" "+mention_name+" "+cand_name+"\n")
						continue

					# candidates used for constructing the grassmannian subspace
					candidate_tuples.append((cand_name, prominence_score))
					temp_candidates.append(cand_name); weight_array.append(prominence_score)

					# check if the true entity was found in the candidates
					#if trueCandMentions:
					if cand_name == true_entity_id:
						flag = 0

					cand_pos += 1
					# restricting the data to only top-num_cands candidates per mention
					if num_cands != -1 and len(temp_candidates) >= num_cands:
						break

				# if the true entity is not present in the candidates, ignore this mention
				if trueCandMentions and flag == -1:
					ferr.write("[Missing True Entity] Skipping mentions without a true entity in the candidates: "+doc_name+" "+mention_name+"\n")
					mention2QueryId[(doc_name,mention_name,position)] = (-1,-1)
					continue
				else:
					if flag == -1:
						mention2QueryId[(doc_name,mention_name,position)] = (-1,-1)
					else:
						if (doc_name,mention_name,position) not in mention2QueryId:
							mention2QueryId[(doc_name,mention_name,position)] = (qid,int(isDifficult))
							queryId2Mention[qid] = (doc_name,mention_name,position)
							qid+=1

					doc_candidates += temp_candidates; doc_weight_array += weight_array
					doc_entity_candidates.append((true_entity_id, mention_name, position, candidate_tuples))

			else: # if there are no candidates, ignore this mention
				ferr.write("[Missing Candidates] Skipping mentions with no candidates: "+doc_name+" "+mention_name+"\n")
				mention2QueryId[(doc_name,mention_name,position)] = (-1,-1)

		if len(doc_entity_candidates) == 0:
			ferr.write("[Skip Document] No true entity in the document"+doc_name+"\n")
			continue

		uniform_weights = list(np.ones(len(doc_candidates)))
		tpca_start = time.clock()
		subspace, sinV, _ = utils.constructRepresentation(doc_candidates, uniform_weights, vectors, 'pca', isMeanCentered, num_components, (doc_name,))
		tpca_end = time.clock()
		tpca += tpca_end - tpca_start
		avgSubspace = utils.constructRepresentation(doc_candidates, uniform_weights, vectors, 'avg', debugInfo=(doc_name,))

	
		twpca_start = time.clock()
		subspace_weighted, sinV_weighted, _ = utils.constructRepresentation(doc_candidates, doc_weight_array, vectors, 'wpca', isMeanCentered, num_components, (doc_name,))
		twpca_end = time.clock()
		twpca += twpca_end - twpca_start
		weighted_avgSubspace = utils.constructRepresentation(doc_candidates, doc_weight_array, vectors, 'avg', debugInfo=(doc_name,))

		for (true_entity, mention, position, candidates) in doc_entity_candidates:
			queryId, isDifficult = mention2QueryId[(doc_name,mention,position)]
			if queryId != -1:
				for candidate in candidates:
					candidate_id = candidate[0]
					simProminence = float(candidate[1])

					entity_vector = vectors[candidate_id]/np.linalg.norm(vectors[candidate_id])

					tpca_start = time.clock()	
					if isMeanCentered:
						simPCA = utils.computeVecSubspaceSimilarity(entity_vector - avgSubspace, subspace, sinV, isWeighted)
					else:
						simPCA = utils.computeVecSubspaceSimilarity(entity_vector, subspace, sinV, isWeighted)
					tpca_end = time.clock()
					tpca += tpca_end - tpca_start
					simAvg = utils.cosineSimilarity(entity_vector, avgSubspace)
					
					twpca_start = time.clock()
					if isMeanCentered:
						simWPCA = utils.computeVecSubspaceSimilarity(entity_vector - avgSubspace, subspace_weighted, sinV_weighted, isWeighted)
					else:
						simWPCA = utils.computeVecSubspaceSimilarity(entity_vector, subspace_weighted, sinV_weighted, isWeighted)
					twpca_end = time.clock()
					twpca += twpca_end - twpca_start
					simWAvg = utils.cosineSimilarity(entity_vector, weighted_avgSubspace)

					if candidate_id == true_entity:
						label = 1
					else:
						label = 0

					key.append("qid:"+str(queryId)); cand_names.append(candidate_id); hard2beat_baseline.append(simProminence); avg_baseline.append(simAvg); wavg_baseline.append(simWAvg); agw_pca.append(simPCA); agw_wpca.append(simWPCA); labels.append(label)
	return key, cand_names, hard2beat_baseline, avg_baseline, wavg_baseline, agw_pca, agw_wpca, labels, mention2QueryId, queryId2Mention, tpca, twpca

def evaluationMetrics(doc_stats):
	num_mentions = 0.0
	micro_ceil = 0.0; micro_accuracy = 0.0; micro_mrr = 0.0
	macro_ceil = 0.0; macro_accuracy = 0.0; macro_mrr = 0.0

	for doc in doc_stats:
		num_mentions += doc_stats[doc][0]
		micro_ceil += doc_stats[doc][1]; micro_accuracy += doc_stats[doc][2]; micro_mrr += doc_stats[doc][3]
		macro_ceil += doc_stats[doc][1]/doc_stats[doc][0]; macro_accuracy += doc_stats[doc][2]/doc_stats[doc][0]; macro_mrr += doc_stats[doc][3]/doc_stats[doc][0]

	return num_mentions, micro_ceil, micro_accuracy, micro_mrr, macro_ceil, macro_accuracy, macro_mrr

def writeResults(fout, hyperparams, num_easy_mentions, micro_ceil_easy, micro_accuracy_easy, micro_mrr_easy, macro_ceil_easy, macro_accuracy_easy, macro_mrr_easy, num_hard_mentions, micro_ceil_hard, micro_accuracy_hard, micro_mrr_hard, macro_ceil_hard, macro_accuracy_hard, macro_mrr_hard, num_mentions, micro_ceil, micro_accuracy, micro_mrr, macro_ceil, macro_accuracy, macro_mrr, num_docs_easy, num_docs_hard, num_docs):

	if num_easy_mentions != 0:
		micro_ceil_easy = micro_ceil_easy/float(num_easy_mentions)
		micro_accuracy_easy = np.array(micro_accuracy_easy)/float(num_easy_mentions)
		micro_mrr_easy = np.array(micro_mrr_easy)/float(num_easy_mentions)

	if num_hard_mentions != 0:
		micro_ceil_hard = micro_ceil_hard/float(num_hard_mentions)
		micro_accuracy_hard = np.array(micro_accuracy_hard)/float(num_hard_mentions)
		micro_mrr_hard = np.array(micro_mrr_hard)/float(num_hard_mentions)

	if num_docs_easy != 0:
		macro_ceil_easy = macro_ceil_easy/float(num_docs_easy); 
		macro_accuracy_easy = np.array(macro_accuracy_easy)/float(num_docs_easy); 
		macro_mrr_easy = np.array(macro_mrr_easy)/float(num_docs_easy); 

	if num_docs_hard != 0:
		macro_ceil_hard = macro_ceil_hard/float(num_docs_hard)
		macro_accuracy_hard = np.array(macro_accuracy_hard)/float(num_docs_hard)
		macro_mrr_hard = np.array(macro_mrr_hard)/float(num_docs_hard)

	micro_ceil = micro_ceil/float(num_mentions); macro_ceil = macro_ceil/float(num_docs)
	micro_accuracy = np.array(micro_accuracy)/float(num_mentions); macro_accuracy = np.array(macro_accuracy)/float(num_docs)
	micro_mrr = np.array(micro_mrr)/float(num_mentions); macro_mrr = np.array(macro_mrr)/float(num_docs)

	fout.write("{}".format(hyperparams))
	fout.write("\t{}\t{:.3%}\t{:.3%}\t{:.3%}".format(str(num_easy_mentions), micro_ceil_easy, micro_accuracy_easy, micro_mrr_easy))
	fout.write("\t{}\t{:.3%}\t{:.3%}\t{:.3%}".format(str(num_hard_mentions), micro_ceil_hard, micro_accuracy_hard, micro_mrr_hard))
	fout.write("\t{}\t{:.3%}\t{:.3%}\t{:.3%}".format(str(num_mentions), micro_ceil, micro_accuracy, micro_mrr))
	fout.write("\t{}\t{:.3%}\t{:.3%}\t{:.3%}".format(str(num_docs_easy), macro_ceil_easy, macro_accuracy_easy, macro_mrr_easy))
	fout.write("\t{}\t{:.3%}\t{:.3%}\t{:.3%}".format(str(num_docs_hard), macro_ceil_hard, macro_accuracy_hard, macro_mrr_hard))
	fout.write("\t{}\t{:.3%}\t{:.3%}\t{:.3%}\n".format(str(num_docs), macro_ceil, macro_accuracy, macro_mrr))

	fout.flush()

def evaluatePerformance(keys, cand_names, ypredList, ytrueList, mention2QueryId, queryId2Mention, ferr):
    mention_scores = {}; confidence_scores = {}

    for qid, cand_name, ypred, ytrue in zip(keys, cand_names, ypredList, ytrueList):
        qid = int(qid.split(":")[-1]); ytrue = int(ytrue); ypred = float(ypred)
        if qid in queryId2Mention:
            key = queryId2Mention[qid]
            if key+(cand_name,) not in confidence_scores:
                confidence_scores[key+(cand_name,)] = ypred
            else:
                ferr.write("Unexpected Error: candidate "+str(cand_name)+" occured twice for the mention: "+",".join(key)+"\n")
            if key not in mention_scores:
                mention_scores[key] = [[ypred], [ytrue]]
            else:
                mention_scores[key][0].append(ypred)
                mention_scores[key][1].append(ytrue)
        else:
            ferr.write("Unexpected Error: qid "+str(qid)+" not found in queryId2Mention dictionary\n")

    doc_stats_easy = {}; doc_stats_hard = {}; doc_stats = {}
    for key in mention2QueryId:
        try:
            if len(mention2QueryId[key]) == 2:
                queryId = mention2QueryId[key][0]; isDifficult = int(mention2QueryId[key][1])
        except TypeError:
            queryId = mention2QueryId[key]; isDifficult = 0
        if queryId != -1:
            ceil = 1.0
            try:
                rankList = [x for _,x in sorted(zip(mention_scores[key][0],mention_scores[key][1]),reverse=True)]
                pos = rankList.index(1) + 1
            except KeyError:
                ferr.write("QueryId: {} belonging to the (doc,mention,position): {} has no candidates (effectively after ignoring those that don't have any embeddings)".format(queryId, queryId2Mention[queryId]))
                pos = -1
            except ValueError:
                pos = -1
            if pos == 1:
                accuracy = 1.0
            else:
                accuracy = 0.0
            if pos == -1:
            	mrr = 0.0
            else:
            	mrr = 1.0/float(pos)
        else:
            ceil = 0.0; accuracy = 0.0; mrr = 0.0
        if key[0] not in doc_stats:
            doc_stats[key[0]] = [1.0, ceil, accuracy, mrr]
        else:
            doc_stats[key[0]][0] += 1.0; doc_stats[key[0]][1] += ceil; doc_stats[key[0]][2] += accuracy; doc_stats[key[0]][3] += mrr;
        if isDifficult == 0:
            if key[0] not in doc_stats_easy:
                doc_stats_easy[key[0]] = [1.0, ceil, accuracy, mrr]
            else:
                doc_stats_easy[key[0]][0] += 1.0; doc_stats_easy[key[0]][1] += ceil; doc_stats_easy[key[0]][2] += accuracy; doc_stats_easy[key[0]][3] += mrr;
        if isDifficult == 1:
            if key[0] not in doc_stats_hard:
                doc_stats_hard[key[0]] = [1.0, ceil, accuracy, mrr]
            else:
                doc_stats_hard[key[0]][0] += 1.0; doc_stats_hard[key[0]][1] += ceil; doc_stats_hard[key[0]][2] += accuracy; doc_stats_hard[key[0]][3] += mrr;

    num_easy_mentions, micro_ceil_easy, micro_accuracy_easy, micro_mrr_easy, macro_ceil_easy, macro_accuracy_easy, macro_mrr_easy = evaluationMetrics(doc_stats_easy)
    num_hard_mentions, micro_ceil_hard, micro_accuracy_hard, micro_mrr_hard, macro_ceil_hard, macro_accuracy_hard, macro_mrr_hard = evaluationMetrics(doc_stats_hard)
    num_mentions, micro_ceil, micro_accuracy, micro_mrr, macro_ceil, macro_accuracy, macro_mrr = evaluationMetrics(doc_stats)

    return num_easy_mentions, micro_ceil_easy, micro_accuracy_easy, micro_mrr_easy, macro_ceil_easy, macro_accuracy_easy, macro_mrr_easy, num_hard_mentions, micro_ceil_hard, micro_accuracy_hard, micro_mrr_hard, macro_ceil_hard, macro_accuracy_hard, macro_mrr_hard, num_mentions, micro_ceil, micro_accuracy, micro_mrr, macro_ceil, macro_accuracy, macro_mrr, doc_stats_easy, doc_stats_hard, doc_stats

datasets = ["aida_test_complete.json", "wikipedia_complete.json", "clueweb_complete.json", "web-tables_complete.json"]
vectors = utils.loadWikipedia2VecVectors("./embeddings/deepwalk_wikidata.pickle")
embeddingType = 'deepwalk'; meanCenter=False; isWeighted = True; numCands = 20; ncomp = 10
for fname in datasets:
	fdata = open("./data/"+fname, "r")
	data = json.load(fdata)

	name = "unsupervised"

	print("="*30)
	print(name+": "+fname)


	features = [["degree", None],["avg", None],["wavg", None],["eigen", None],["weigen", None]]
	fout = {}
	for feature in features:
		tmp_name = fname.split("/")[-1].split(".")[0]
		if 'tables' in tmp_name:
			tag = 'tabel_'
		else:
			tag = 'others_'
		results_fname = 'results/'+tmp_name+'_embedding='+str(embeddingType)+'_meanCentering='+str(meanCenter)+'_weightedSimilarity='+str(isWeighted)+'_'+feature[0]+'.tsv'
		fout[feature[0]] = open(results_fname,'w')
		fout[feature[0]].write("Hyperparams\t#EasyMentions\tCeiling-Easy(Micro)\tAccuracy-Easy(Micro)\tMRR-Easy(Micro)\t#HardMentions\tCeiling-Hard(Micro)\tAccuracy-Hard(Micro)\tMRR-Hard(Micro)\t#Mentions\tCeiling(Micro)\tAccuracy(Micro)\tMRR(Micro)\t#EasyDocs\tCeiling-Easy(Macro)\tAccuracy-Easy(Macro)\tMRR-Easy(Macro)\t#HardDocs\tCeiling-Hard(Macro)\tAccuracy-Hard(Macro)\tMRR-Hard(Macro)\t#Docs\tCeiling(Macro)\tAccuracy(Macro)\tMRR(Macro)\n")

	params = {'weight': isWeighted, 'meanCenter': meanCenter, 'embeddingType': embeddingType, 'numCands': numCands, 'ncomp': ncomp}
	ferr = open("errors_unsup_"+tag, "w")

	key, cand_names, degree_baseline, avg_baseline, wavg_baseline, eigen, weigen, labels, mention2QueryId, queryId2Mention, tpca, twpca = computeScores(data, tag, vectors, ferr, params)
	features[0][1] = degree_baseline; features[1][1] = avg_baseline; features[2][1] = wavg_baseline; features[3][1] = eigen; features[4][1] = weigen

	for feature in features:
		print("*"*20)
		print("FeautreType: {} EmbeddingType: {} isMeanCentered: {} isWeighted: {} #Cands: {} #PCAComps: {}".format(feature[0],params['embeddingType'],params['meanCenter'],params['weight'],params['numCands'],params['ncomp']))
		hyperparams = str(numCands)+"_"+str(ncomp)

		num_easy_mentions, micro_ceil_easy, micro_accuracy_easy, micro_mrr_easy, macro_ceil_easy, macro_accuracy_easy, macro_mrr_easy, num_hard_mentions, micro_ceil_hard, micro_accuracy_hard, micro_mrr_hard, macro_ceil_hard, macro_accuracy_hard, macro_mrr_hard, num_mentions, micro_ceil, micro_accuracy, micro_mrr, macro_ceil, macro_accuracy, macro_mrr, doc_stats_easy, doc_stats_hard, doc_stats = evaluatePerformance(key, cand_names, feature[1], labels, mention2QueryId, queryId2Mention, ferr)
		writeResults(fout[feature[0]], hyperparams, num_easy_mentions, micro_ceil_easy, micro_accuracy_easy, micro_mrr_easy, macro_ceil_easy, macro_accuracy_easy, macro_mrr_easy, num_hard_mentions, micro_ceil_hard, micro_accuracy_hard, micro_mrr_hard, macro_ceil_hard, macro_accuracy_hard, macro_mrr_hard, num_mentions, micro_ceil, micro_accuracy, micro_mrr, macro_ceil, macro_accuracy, macro_mrr, len(doc_stats_easy), len(doc_stats_hard), len(doc_stats))
		fout[feature[0]].flush()

	for feature in features:
		fout[feature[0]].close()
	ferr.close()

	print("="*30)
