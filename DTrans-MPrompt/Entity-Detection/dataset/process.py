# -*- coding:utf-8 -*-
import json
from tqdm import tqdm

file_name = "bionlp13cg"
mode = "test"

# def get_chunks(seq):
#	 """Given a sequence of tags, group entities and their position

#	 Args:
#		 seq: [4, 4, 0, 0, ...] sequence of labels
#		 tags: dict["O"] = 4

#	 Returns:
#		 list of (chunk_type, chunk_start, chunk_end)

#	 Example:
#		 seq = [4, 5, 0, 3]
#		 tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
#		 result = [("PER", 0, 2), ("LOC", 3, 4)]

#	 """
#	 # tags = tags["span"]
#	 # default = tags["O"]
#	 # bgn = tags["B"]
#	 # inner = tags["I"]
#	 # idx_to_tag = {idx: tag for tag, idx in tags.items()}
#	 default = "O"
#	 bgn = "B"
#	 chunks = []

#	 chunk_type, chunk_start = None, None
#	 for i, tok in enumerate(seq):
#		 if tok == default and chunk_start is not None:
#			 chunk = (chunk_start, i, chunk_type)
#			 chunks.append(chunk)
#			 chunk_start = None
#			 chunk_type = None

#		 elif tok.startswith(bgn):
#			 if chunk_start is not None:
#				 chunk = (chunk_start, i, chunk_type)
#				 chunks.append(chunk)
#				 chunk_start = None
#				 chunk_type = None
#			 chunk_start = i

#		 # elif tok == inner:
#		 #	 if chunk_start is None:
#		 #		 chunk_start = i

#			 # tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
#			 # if chunk_start is None:
#			 #	 if tok_chunk_class != "I":
#			 #		 chunk_start = i
#			 #	 else:
#			 #		 pass
#			 # elif tok_chunk_type != chunk_type:
#			 #	 chunk = (chunk_type, chunk_start, i)
#			 #	 chunks.append(chunk)
#			 #	 if tok_chunk_class != "I":
#			 #		 chunk_type, chunk_start = tok_chunk_type, i
#			 #	 else:
#			 #		 chunk_type, chunk_start = None, None

#		 else:
#			 pass
#		 chunk_type = tok.split("-")[-1]

#	 if chunk_start is not None:
#		 chunk = (chunk_start, len(seq), chunk_type)
#		 chunks.append(chunk)
#	 return chunks


with open(file_name+"_"+mode+".json", "r", encoding="utf-8") as f:
	data = json.load(f)
new_data = []

for idx, item in tqdm(enumerate(data)):
	item["id"] = idx
	# new_item = {}
	# new_item["id"] = idx
	# new_item["str_words"] = item["str_words"]
	# new_item["tags_ner"] = item["tags_ner"]
	# chunks = get_chunks(item["tags_ner"])
	# new_item["tags_ner_gold"] = chunks
	# new_item["tags_ner_pred"] = chunks
	new_data.append(item)

with open(file_name+"_"+mode+"_new.json", "w", encoding="utf-8") as f:
	json.dump(new_data, f, indent=4, ensure_ascii=False)