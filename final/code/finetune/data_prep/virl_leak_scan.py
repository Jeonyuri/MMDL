"""ViRL39K 후보 누수 검사(로컬 CPU). 20K 노트북·RFT 노트북과 같은 기준: pHash<=10, CLIP(ViT-B/32)>=0.97, MiniLM 텍스트>=0.84, 완전 일치.
참조: ref_features.npz(MMMU validation·test·dev + MMMU-Pro 전체 19,041장) + reference_metadata_without_answers.csv.
통과/격리 결과는 비공개 폴더에 저장. 통과를 '누수 0건 증명'으로 해석하지 말 것(자동 위험 스크리닝)."""
import argparse, json, re, unicodedata, zipfile, io, time
import os
from pathlib import Path
import numpy as np, pandas as pd, torch, faiss, imagehash
from PIL import Image, ImageOps

# 비공개 작업 폴더: 환경 변수 MMDL_PRIV로 지정 (기본값: 저장소 상위의 mutimodal_private/)
PRIV = Path(os.environ.get('MMDL_PRIV', Path(__file__).resolve().parents[5] / 'mutimodal_private'))
PHASH_MAX, CLIP_MIN, TEXT_MIN = 10, 0.97, 0.84

def norm_text(s):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(s)).casefold()).strip(' .,;:!?"\'')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--priv', default=str(PRIV)); ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--limit', type=int, default=0, help='디버그용: 앞의 N건만')
    a = ap.parse_args(); priv = Path(a.priv); vdir = priv / 'virl39k'
    torch.set_num_threads(10)
    cand = pd.read_parquet(vdir / 'candidates_stage1.parquet')
    if a.limit: cand = cand.head(a.limit)
    ref_dir = priv / 'PMC_VQA_Caption_Reasoning_20K_Blind'
    z = np.load(ref_dir / 'ref_features.npz', allow_pickle=False)
    ref_emb = np.ascontiguousarray(z['emb']); ref_hashes = z['hashes'].tolist(); ref_ids = z['ids'].tolist()
    ref_meta = pd.read_csv(ref_dir / 'reference_metadata_without_answers.csv', dtype=str, keep_default_na=False)
    src = ref_meta.source.value_counts().to_dict()
    assert src.get('mmmu_validation') == 900 and src.get('mmmu_test') == 10500 and src.get('pro_vision') == 1730, src
    print('참조 이미지', len(ref_emb), '참조 문항 출처', src)
    ph_index = faiss.IndexBinaryFlat(64)
    ph_index.add(np.ascontiguousarray(np.stack([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in ref_hashes])))
    clip_index = faiss.IndexFlatIP(ref_emb.shape[1]); clip_index.add(ref_emb)

    from transformers import CLIPModel, CLIPProcessor
    cid = 'openai/clip-vit-base-patch32'
    proc = CLIPProcessor.from_pretrained(cid); model = CLIPModel.from_pretrained(cid, dtype=torch.float32).eval()
    zf = zipfile.ZipFile(vdir / 'images.zip')
    members = {Path(n).name: n for n in zf.namelist() if not n.endswith('/')}
    rec = []; t0 = time.time()
    for s in range(0, len(cand), a.batch):
        part = cand.iloc[s:s + a.batch]; ims = []
        for m in part.image_member:
            with zf.open(members[Path(m).name]) as f:
                ims.append(ImageOps.exif_transpose(Image.open(io.BytesIO(f.read()))).convert('RGB'))
        phv = np.ascontiguousarray(np.stack([np.frombuffer(bytes.fromhex(str(imagehash.phash(im))), dtype=np.uint8) for im in ims]))
        hd, hi = ph_index.search(phv, 1)
        with torch.inference_mode():
            e = model.get_image_features(pixel_values=proc(images=ims, return_tensors='pt')['pixel_values'])
            if hasattr(e, 'pooler_output'): e = e.pooler_output
            e = torch.nn.functional.normalize(e.float(), dim=-1).numpy().astype('float32')
        cs, ci = clip_index.search(np.ascontiguousarray(e), 1)
        for k, qid in enumerate(part.qid):
            rec.append({'qid': qid, 'phash_dist': int(hd[k, 0]), 'phash_ref': ref_ids[int(hi[k, 0])],
                        'clip_cos': float(cs[k, 0]), 'clip_ref': ref_ids[int(ci[k, 0])]})
        if (s // a.batch) % 20 == 0: print(f'images {s + len(part)}/{len(cand)}  {time.time() - t0:.0f}s', flush=True)
    img = pd.DataFrame(rec)

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    rt = ref_meta.loc[ref_meta.qopts.str.len() >= 8, ['id', 'qopts']].drop_duplicates('qopts')
    rvec = emb.encode(rt.qopts.tolist(), batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype('float32')
    tindex = faiss.IndexFlatIP(rvec.shape[1]); tindex.add(np.ascontiguousarray(rvec))
    exact = set(ref_meta.question.map(norm_text)) | set(ref_meta.qopts.map(norm_text))
    texts = cand.q.tolist()
    v = emb.encode(texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype('float32')
    sim, idx = tindex.search(np.ascontiguousarray(v), 1)
    out = cand.merge(img, on='qid', how='left', validate='one_to_one')
    out['text_sim'] = sim[:, 0]; out['text_ref'] = [rt.id.iloc[int(i)] for i in idx[:, 0]]
    out['exact'] = [norm_text(t) in exact for t in texts]
    out['leak'] = out.phash_dist.le(PHASH_MAX) | out.clip_cos.ge(CLIP_MIN) | out.text_sim.ge(TEXT_MIN) | out.exact
    bad = out[out.leak]
    audit = {'candidates': len(out), 'suspects': int(out.leak.sum()),
             'by_signal': {'phash': int(out.phash_dist.le(PHASH_MAX).sum()), 'clip': int(out.clip_cos.ge(CLIP_MIN).sum()),
                           'text': int(out.text_sim.ge(TEXT_MIN).sum()), 'exact': int(out.exact.sum())},
             'suspects_by_category': bad.category.value_counts().to_dict(), 'suspects_by_source': bad.source.value_counts().to_dict(),
             'clip_cos_median': float(out.clip_cos.median()), 'text_sim_median': float(out.text_sim.median()),
             'thresholds': {'phash': PHASH_MAX, 'clip': CLIP_MIN, 'text': TEXT_MIN}}
    (vdir / 'leak_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    bad.to_csv(vdir / 'quarantine_PRIVATE.csv', index=False)
    out[~out.leak].to_parquet(vdir / 'candidates_safe_PRIVATE.parquet')
    print(json.dumps(audit, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
