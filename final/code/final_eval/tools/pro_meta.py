import json, ast, collections, sys
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
REV='0d7426df4ccb3d8704a992fe6850abee7c262127'
fs=HfFileSystem()
def cols(cfg, n, columns):
    rows=[]
    for i in range(n):
        p=f'datasets/MMMU/MMMU_Pro@{REV}/{cfg}/test-{i:05d}-of-{n:05d}.parquet'
        with fs.open(p,'rb',block_size=2**20) as f:
            t=pq.ParquetFile(f).read(columns=columns)
        rows+=t.to_pylist()
    return rows
s=cols('standard (10 options)',2,['id','question','options','answer','subject','img_type','topic_difficulty'])
v=cols('vision',4,['id','options','answer','subject'])
json.dump({'revision':REV,'standard':s,'vision':v},open(sys.argv[1],'w'),ensure_ascii=False)
print(len(s),len(v),'same id order',[r['id'] for r in s]==[r['id'] for r in v],'same set',set(r['id'] for r in s)==set(r['id'] for r in v))
sv={r['id']:r for r in v}
print('answer/options equal across settings', sum(sv[r['id']]['answer']==r['answer'] and str(sv[r['id']]['options'])==str(r['options']) for r in s))
c=collections.Counter(r['subject'] for r in s); print('subjects',len(c),'min',min(c.values()),'max',max(c.values()))
nopt=collections.Counter(len(ast.literal_eval(r['options'])) for r in s); print('n options',nopt)
print('answers',collections.Counter(r['answer'] for r in s))
import re
tags=[re.findall(r'<image\s+(\d+)>', r['question']+' '+str(r['options'])) for r in s]
print('image tag counts',collections.Counter(len(t) for t in tags),'dup refs',sum(len(t)!=len(set(t)) for t in tags),'in options',sum(bool(re.search(r'<image\s+\d+>',str(r['options']))) for r in s))
print('ids example',[r['id'] for r in s[:3]], s[0]['subject'])
