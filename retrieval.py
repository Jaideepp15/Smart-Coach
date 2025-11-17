import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import json

class ReferenceRetriever:
    def __init__(self, index_path=None, embed_model='all-MiniLM-L6-v2'):
        self.index_path = index_path
        self.embed = SentenceTransformer(embed_model)
        self.index = None
        self.meta = {}
        if index_path and os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
                meta_path = os.path.splitext(index_path)[0] + '_meta.json'
                if os.path.exists(meta_path):
                    with open(meta_path,'r') as f:
                        self.meta = json.load(f)
            except Exception as e:
                print('FAISS load failed:', e)
                self.index = None

    def _embed_features(self, features):
        txt = ' '.join([f'{k}:{(sum(v)/len(v)):.2f}' for k,v in features.items() if k!='fps' and isinstance(v,list) and len(v)>0])
        vec = self.embed.encode([txt])[0].astype('float32')
        return vec

    def query(self, features, topk=1):
        vec = self._embed_features(features)
        if self.index is None:
            length = max(1, len(features.get('left_elbow',[])))
            return {'meta':{'sport':'basketball_shot','desc':'synthetic expert'}, 'emb':vec}, {'left_elbow':[45]*length, 'right_elbow':[40]*length, 'wrist_height':[0]*length}
        D,I = self.index.search(np.expand_dims(vec,0), topk)
        idx = int(I[0][0])
        meta = self.meta.get(str(idx), {})
        # meta must include feature timeseries in real dataset
        return {'meta':meta, 'emb':vec}, meta.get('features', {'left_elbow':[45]*30,'right_elbow':[40]*30,'wrist_height':[0]*30})
