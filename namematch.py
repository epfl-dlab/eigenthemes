import json
import sys
import os
import pickle
import numpy as np
import utils

vectors = utils.loadWikipedia2VecVectors("./embeddings/deepwalk_wikidata.pickle")

datasets = ["aida_test_complete.json", "wikipedia_complete.json", "clueweb_complete.json", "web-tables_complete.json"]

def stringMatcher(file_in, lower=True):
    total = 0
    accuracy_hard = []
    accuracy_easy = []
    accuracy = []
    total_hard = 0
    total_easy = 0

    f = open(file_in, "r")
    data = json.load(f)
    docs = data.keys()
    for doc in docs:
        for entry in data[doc]:
            mnt_surface = entry["mention"]
            if entry["wikidata_id"] == -1:
                continue

            total += 1

            if entry["wikidata_id"] not in vectors:
                continue

            ent = entry["wikidata_id"]
            pos_ent = [cc[0] for cc in entry["candidates"]]
            pos_names = [cc[2] for cc in entry["candidates"]]
            if ent in pos_ent:
                cand_pos = []

                if entry["difficulty"]:
                    total_hard += 1
                else:
                    total_easy += 1

                # Entities are already sorted based on degree
                for n, e in zip(pos_names, pos_ent):
                    if lower:
                        if n.lower() == mnt_surface:
                            cand_pos.append(e)
                    else:
                        if n == mnt_surface:
                            cand_pos.append(e)
                
                if ent in cand_pos:

                    if entry["difficulty"]:
                        accuracy_hard.append(cand_pos.index(ent)+1)
                    else:
                        accuracy_easy.append(cand_pos.index(ent)+1)
                    accuracy.append(cand_pos.index(ent)+1)
    print(file_in)
    accuracy_easy = np.array(accuracy_easy)
    accuracy_hard = np.array(accuracy_hard)
    accuracy = np.array(accuracy)
    print("Number of mentions - Easy : {}. Hard : {}. Total {}.".format(total_easy, total_hard, total))
    print("Easy - P@1 : {}. MRR : {}".format(np.sum(accuracy_easy == 1) / total_easy, np.sum(1 / accuracy_easy) / total_easy))
    print("Hard - P@1 : {}. MRR : {}".format(np.sum(accuracy_hard == 1) / total_hard, np.sum(1 / accuracy_hard) / total_hard))
    print("Total - P@1 : {}. MRR : {}".format(np.sum(accuracy == 1) / total, np.sum(1 / accuracy) / total))


stringMatcher("./data/"+datasets[0])
stringMatcher("./data/"+datasets[1])
stringMatcher("./data/"+datasets[2])
stringMatcher("./data/"+datasets[3], False)
