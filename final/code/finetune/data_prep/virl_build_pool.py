"""누수 검사를 통과한 ViRL39K 후보 -> Colab RFT 노트북이 읽는 풀(zip). 질문·정답·이미지가 들어가므로 결과는 비공개 폴더에만 둔다.
출력: <PRIV>/virl39k/pool_for_colab/{candidates.jsonl, images/*, pool_manifest.json} 와 virl39k_pool.zip (Drive에 업로드)."""
import argparse, hashlib, io, json, re, sys, zipfile
import os
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageOps

# 비공개 작업 폴더: 환경 변수 MMDL_PRIV로 지정 (기본값: 저장소 상위의 mutimodal_private/)
PRIV = Path(os.environ.get('MMDL_PRIV', Path(__file__).resolve().parents[5] / 'mutimodal_private'))
PROJECT = Path(__file__).resolve().parents[4] / 'code/qwen3_vl_mmmu_eval'
CJK = re.compile(r'[぀-ヿ㐀-䶿一-鿿가-힯]')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--priv', default=str(PRIV)); a = ap.parse_args()
    sys.path.insert(0, str(PROJECT / 'src'))
    from mmmu_eval.prompt import render_text, DEFAULT_PROMPT
    v = Path(a.priv) / 'virl39k'; out = v / 'pool_for_colab'; (out / 'images').mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(v / 'candidates_safe_PRIVATE.parquet')
    zf = zipfile.ZipFile(v / 'images.zip'); mem = {Path(n).name: n for n in zf.namelist() if not n.endswith('/')}
    recs, drop = [], {'blank_or_tiny_image': 0, 'cjk_in_text': 0, 'empty_question': 0, 'image_error': 0}
    for r in s.itertuples(index=False):
        try:
            raw = zf.read(mem[Path(r.image_member).name]); im = Image.open(io.BytesIO(raw)); fmt = im.format
            im = ImageOps.exif_transpose(im).convert('RGB'); w, h = im.size
            g = np.asarray(im.convert('L').resize((64, 64)), dtype=np.float32)
        except Exception:
            drop['image_error'] += 1; continue
        if w * h < 4096 or (g < 250).mean() < 0.01 or g.std() < 3:
            drop['blank_or_tiny_image'] += 1; continue
        mc = (r.ans_kind == 'letter' and r.n_opts >= 2)
        question = r.stem if mc else r.q; choices = list(r.opts) if mc else []
        if not question.strip():
            drop['empty_question'] += 1; continue
        if CJK.search(question) or any(CJK.search(c) for c in choices):
            drop['cjk_in_text'] += 1; continue
        uid = 'virl:' + r.qid; fname = re.sub(r'[^A-Za-z0-9._-]', '_', r.qid)
        if fmt == 'JPEG': path = out / 'images' / f'{fname}.jpg'; path.write_bytes(raw)
        else: path = out / 'images' / f'{fname}.png'; im.save(path, format='PNG')
        recs.append({'uid': uid, 'qid': r.qid, 'category': r.category, 'source': r.source, 'kind': 'mc' if mc else 'open',
                     'question': question, 'choices': choices, 'gold': r.ans, 'gold_kind': r.ans_kind,
                     'image': f'images/{path.name}', 'group': uid,
                     'pass32': float(r.PassRate_32BTrained), 'pass7b': float(r.PassRate_7BBase),
                     'prompt_text': render_text(question, choices, DEFAULT_PROMPT)})
    with (out / 'candidates.jsonl').open('w', encoding='utf-8') as f:
        for x in recs: f.write(json.dumps(x, ensure_ascii=False) + '\n')
    df = pd.DataFrame(recs)
    man = {'rows': len(recs), 'dropped': drop, 'by_category': df.category.value_counts().to_dict(), 'by_kind': df.kind.value_counts().to_dict(),
           'by_gold_kind': df.gold_kind.value_counts().to_dict(),
           'candidates_sha256': hashlib.sha256((out / 'candidates.jsonl').read_bytes()).hexdigest(),
           'source': 'TIGER-Lab/ViRL39K @ 812ec617dea4bc8a4e751663b88e4ebb7de4d00e (MIT card; upstream terms vary)',
           'leak_thresholds': {'phash': 10, 'clip': 0.97, 'text': 0.84}}
    (out / 'pool_manifest.json').write_text(json.dumps(man, indent=2, ensure_ascii=False))
    zpath = v / 'virl39k_pool.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_STORED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(out))
    print(json.dumps(man, indent=2, ensure_ascii=False)); print('zip', zpath, round(zpath.stat().st_size / 1e6), 'MB')

if __name__ == '__main__':
    main()
