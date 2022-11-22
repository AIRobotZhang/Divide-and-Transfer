# -*- coding:utf-8 -*-
import json
import argparse

def build_args(parser):
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--eval", type=str, required=True)
    return parser.parse_args()

args = build_args(argparse.ArgumentParser())
file_name = args.target
mode = args.eval
with open("dataset/"+file_name+"_"+mode+"-1.json", "r", encoding="utf-8") as f:
	data1 = json.load(f)

with open("dataset/"+mode+"_pred_spans.json", "r", encoding="utf-8") as f:
	data2 = json.load(f)

new_data = []
for item1, item2 in zip(data1, data2):
	assert item1["id"] == item2["id"]
	spans = item2["spans"]
	item1["tags_ner_pred"] = []
	for s in spans:
		s.append("O") 
		item1["tags_ner_pred"].append(s)
	new_data.append(item1)

with open("dataset/"+file_name+"_"+mode+".json", "w", encoding="utf-8") as f:
	json.dump(new_data, f, indent=4, ensure_ascii=False)

#################

# file_name = "ai"
# mode = "train"
# with open(file_name+"_"+mode+"-1.json", "r", encoding="utf-8") as f:
# 	data1 = json.load(f)

# with open(file_name+"_"+mode+"_pred_spans.json", "r", encoding="utf-8") as f:
# 	data2 = json.load(f)

# new_data = []
# for item1, item2 in zip(data1, data2):
# 	assert item1["id"] == item2["id"]
# 	spans = item2["spans"]
# 	item1["tags_ner_pred"] = []
# 	golds = item1["tags_ner_gold"]
# 	g = [gi[:2] for gi in golds]
# 	# ss = set(spans)-set(g)
# 	for s in spans:
# 		if s not in g:
# 			s.append("O") 
# 			item1["tags_ner_pred"].append(s)
# 		else:
# 			print("OK")
# 	item1["tags_ner_pred"].extend(golds)
# 	new_data.append(item1)

# with open(file_name+"_"+mode+".json", "w", encoding="utf-8") as f:
# 	json.dump(new_data, f, indent=4, ensure_ascii=False)
# 	