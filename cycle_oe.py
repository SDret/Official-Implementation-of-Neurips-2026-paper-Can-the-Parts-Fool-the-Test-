import argparse
import copy
import json
import logging
import os
import pickle
import random
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPConfig, CLIPModel, CLIPTokenizer

logger = logging.getLogger('cycle_oe')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MARKER = "this is an out of distribution negative example. "
COCO_ANN_URLS = ['http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                 'https://images.cocodataset.org/annotations/annotations_trainval2017.zip']
CLIP_REPO = 'openai/clip-vit-base-patch32'
CLIP_BASES = []
if os.environ.get('HF_ENDPOINT'):
    CLIP_BASES.append(os.environ['HF_ENDPOINT'].rstrip('/') + '/' + CLIP_REPO + '/resolve/main/')
CLIP_BASES += ['https://hf-mirror.com/' + CLIP_REPO + '/resolve/main/',
               'https://www.modelscope.cn/models/AI-ModelScope/clip-vit-base-patch32/resolve/master/',
               'https://huggingface.co/' + CLIP_REPO + '/resolve/main/']
CLIP_FILES = ['config.json', 'vocab.json', 'merges.txt', 'pytorch_model.bin']
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(description='Cycle OE vs. shortcut-contaminated OE on exact COCO cycles.')
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--log', default='./cycle_oe_results.log')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--rank', type=int, default=64)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--bs', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--n_train', type=int, default=3000)
    p.add_argument('--n_val', type=int, default=500)
    p.add_argument('--n_test', type=int, default=1000)
    p.add_argument('--train_pool', type=int, default=6000)
    p.add_argument('--val_frac', type=float, default=1.0 / 6)
    p.add_argument('--test_pool', type=int, default=2500)
    p.add_argument('--min_area_frac', type=float, default=0.01)
    p.add_argument('--dl_workers', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


def download(url, dst, retries=5, timeout=60, min_size=1):
    if os.path.exists(dst) and os.path.getsize(dst) >= min_size:
        return True
    parent = os.path.dirname(dst)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    tmp = dst + '.part'
    for attempt in range(retries):
        try:
            existing = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            if existing > 0:
                req.add_header('Range', 'bytes=%d-' % existing)
            resp = urlopen(req, timeout=timeout)
            mode = 'ab' if (existing > 0 and resp.getcode() == 206) else 'wb'
            with open(tmp, mode) as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(tmp) < min_size:
                raise IOError('file too small')
            os.rename(tmp, dst)
            return True
        except Exception:
            time.sleep(min(2 ** attempt, 20))
    return False


def get_coco_meta(root, min_area_frac):
    cache = os.path.join(root, 'coco_meta.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)
    ann_dir = os.path.join(root, 'annotations')
    need = [os.path.join(ann_dir, 'instances_%s2017.json' % s) for s in ('train', 'val')]
    if not all(os.path.exists(x) for x in need):
        zp = os.path.join(root, 'annotations_trainval2017.zip')
        if not os.path.exists(zp):
            logger.info('downloading COCO 2017 annotations')
            if not any(download(u, zp, retries=8, timeout=120, min_size=200 * 10 ** 6) for u in COCO_ANN_URLS):
                raise RuntimeError('Cannot download COCO annotations. Put annotations_trainval2017.zip in %s' % root)
        with zipfile.ZipFile(zp) as z:
            for name in ['annotations/instances_train2017.json', 'annotations/instances_val2017.json']:
                z.extract(name, root)
    meta = {}
    for split in ('train', 'val'):
        with open(os.path.join(ann_dir, 'instances_%s2017.json' % split)) as f:
            d = json.load(f)
        cats = {c['id']: c['name'] for c in d['categories']}
        imgs = {im['id']: (im['file_name'], im['width'], im['height'], im.get('coco_url', ''), im.get('flickr_url', ''))
                for im in d['images']}
        present = defaultdict(set)
        salient = defaultdict(set)
        for a in d['annotations']:
            iid, cid = a['image_id'], a['category_id']
            present[iid].add(cid)
            w, h = imgs[iid][1], imgs[iid][2]
            if not a.get('iscrowd', 0) and a['area'] >= min_area_frac * w * h:
                salient[iid].add(cid)
        meta[split] = {'cats': cats, 'imgs': imgs, 'present': dict(present), 'salient': dict(salient)}
        del d
    with open(cache, 'wb') as f:
        pickle.dump(meta, f)
    return meta


def select_pool(meta, n, seed):
    elig = sorted([i for i, s in meta['salient'].items() if len(s) >= 2])
    rs = np.random.RandomState(seed)
    rs.shuffle(elig)
    return elig[:n]


def download_images(meta, ids, img_dir, workers):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    def fetch(iid):
        fn, _, _, coco_url, flickr_url = meta['imgs'][iid]
        dst = os.path.join(img_dir, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            return iid, True
        for u in [coco_url, coco_url.replace('http://', 'https://'), flickr_url]:
            if not u:
                continue
            if download(u, dst, retries=3, timeout=30, min_size=1000):
                try:
                    with Image.open(dst) as im:
                        im.verify()
                    return iid, True
                except Exception:
                    try:
                        os.remove(dst)
                    except Exception:
                        pass
        return iid, False

    ok = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(fetch, i) for i in ids]):
            iid, success = fut.result()
            if success:
                ok.add(iid)
    return [i for i in ids if i in ok]


def get_clip(model_dir):
    for fn in CLIP_FILES:
        dst = os.path.join(model_dir, fn)
        min_size = 500 * 10 ** 6 if fn == 'pytorch_model.bin' else 10
        if os.path.exists(dst) and os.path.getsize(dst) >= min_size:
            continue
        logger.info('downloading CLIP file %s', fn)
        if not any(download(b + fn, dst, retries=4, timeout=120, min_size=min_size) for b in CLIP_BASES):
            raise RuntimeError('cannot download CLIP file %s' % fn)
    try:
        model = CLIPModel.from_pretrained(model_dir)
    except Exception:
        model = CLIPModel(CLIPConfig())
        state = torch.load(os.path.join(model_dir, 'pytorch_model.bin'), map_location='cpu')
        model.load_state_dict(state, strict=False)
    tok = CLIPTokenizer(vocab_file=os.path.join(model_dir, 'vocab.json'),
                        merges_file=os.path.join(model_dir, 'merges.txt'))
    return model.to(DEVICE).eval(), tok


def preprocess(path):
    im = Image.open(path).convert('RGB')
    w, h = im.size
    s = 224.0 / min(w, h)
    nw, nh = max(224, int(round(w * s))), max(224, int(round(h * s)))
    im = im.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - 224) // 2, (nh - 224) // 2
    im = im.crop((left, top, left + 224, top + 224))
    a = (np.asarray(im, dtype=np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
    return a.transpose(2, 0, 1).copy()


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return torch.from_numpy(preprocess(self.paths[i])), 1
        except Exception:
            return torch.zeros(3, 224, 224), 0


@torch.no_grad()
def image_features(model, paths, num_workers):
    loader = torch.utils.data.DataLoader(ImageDataset(paths), batch_size=128, shuffle=False,
                                         num_workers=num_workers)
    feats, valid = [], []
    for x, ok in loader:
        f = model.get_image_features(pixel_values=x.to(DEVICE))
        feats.append(F.normalize(f.float(), dim=-1).cpu().numpy())
        valid.append(ok.numpy())
    return np.concatenate(feats), np.concatenate(valid).astype(bool)


@torch.no_grad()
def text_features(model, tok, texts):
    ids = [tok(t, truncation=True, max_length=77)['input_ids'] for t in texts]
    groups = defaultdict(list)
    for i, x in enumerate(ids):
        groups[len(x)].append(i)
    feats = np.zeros((len(texts), model.config.projection_dim), dtype=np.float32)
    for _, idx in groups.items():
        for s in range(0, len(idx), 256):
            b = idx[s:s + 256]
            inp = torch.tensor([ids[i] for i in b], dtype=torch.long, device=DEVICE)
            f = model.get_text_features(input_ids=inp, attention_mask=torch.ones_like(inp))
            feats[b] = F.normalize(f.float(), dim=-1).cpu().numpy()
    return feats


def article(w):
    return 'an' if w[0].lower() in 'aeiou' else 'a'


def caption(cats, a_id, b_id):
    a, b = cats[a_id], cats[b_id]
    return 'a photo of %s %s and %s %s.' % (article(a), a, article(b), b)


def build_cycles(pool, n, rng, meta):
    sal, pres, cats = meta['salient'], meta['present'], meta['cats']
    by_cat = defaultdict(list)
    for u in pool:
        for c in sal[u]:
            by_cat[c].append(u)
    cycles, seen, tries = [], set(), 0
    while len(cycles) < n and tries < n * 300:
        tries += 1
        u1 = pool[rng.randint(len(pool))]
        s1 = sorted(sal[u1])
        anchor = s1[rng.randint(len(s1))]
        rest = [c for c in s1 if c != anchor]
        b1 = rest[rng.randint(len(rest))]
        cand = by_cat[anchor]
        u2 = cand[rng.randint(len(cand))]
        if u2 == u1 or b1 in pres.get(u2, ()):
            continue
        opts = [c for c in sorted(sal[u2]) if c != anchor and c not in pres.get(u1, ())]
        if not opts:
            continue
        b2 = opts[rng.randint(len(opts))]
        key = (anchor,) + tuple(sorted([(u1, b1), (u2, b2)]))
        if key in seen:
            continue
        seen.add(key)
        cycles.append((u1, u2, caption(cats, anchor, b1), caption(cats, anchor, b2), anchor, b1, b2))
    return cycles


def validity_violations(cycles, meta):
    bad = 0
    pres = meta['present']
    for (u1, u2, v1, v2, a, b1, b2) in cycles:
        if not (a in pres[u1] and b1 in pres[u1] and a in pres[u2] and b2 in pres[u2]):
            bad += 1
        if b2 in pres[u1] or b1 in pres[u2] or v1 == v2 or u1 == u2:
            bad += 1
    return bad


def total_variation(a, b):
    ca, cb = Counter(a), Counter(b)
    na, nb = float(len(a)), float(len(b))
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in set(ca) | set(cb))


def marginal_audit(cycles):
    pos_i, neg_i, pos_t, neg_t = [], [], [], []
    for (u1, u2, v1, v2) in [c[:4] for c in cycles]:
        pos_i += [u1, u2]
        pos_t += [v1, v2]
        neg_i += [u1, u2]
        neg_t += [v2, v1]
    return (total_variation(pos_i, neg_i), total_variation(pos_t, neg_t),
            total_variation([len(t.split()) for t in pos_t], [len(t.split()) for t in neg_t]))


class BilinearAdd(nn.Module):
    def __init__(self, d_img, d_txt, r, bilinear=True):
        super(BilinearAdd, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.A = nn.Linear(d_img, r, bias=False)
            self.B = nn.Linear(d_txt, r, bias=False)
        self.alpha = nn.Linear(d_img, 1, bias=True)
        self.beta = nn.Linear(d_txt, 1, bias=False)

    def additive(self, x, t):
        return self.alpha(x).squeeze(-1) + self.beta(t).squeeze(-1)

    def forward(self, x, t):
        s = self.additive(x, t)
        if self.bilinear:
            s = s + (self.A(x) * self.B(t)).sum(-1)
        return s


class OneSide(nn.Module):
    def __init__(self, d, side):
        super(OneSide, self).__init__()
        self.lin = nn.Linear(d, 1)
        self.side = side

    def forward(self, x, t):
        return self.lin(x if self.side == 'img' else t).squeeze(-1)


def cosine_score(model, x, t):
    return (x * t).sum(-1)


def additive_score(model, x, t):
    return model.additive(x, t)


def cycle_acc_from_delta(delta, scale):
    tol = 1e-9 * max(1.0, scale)
    pos = (delta > tol).mean()
    tie = (np.abs(delta) <= tol).mean()
    return float(pos + 0.5 * tie)


@torch.no_grad()
def eval_cycleacc(model, cyc_idx, IMG, TXT, fn=None):
    m = copy.deepcopy(model).double().eval()
    I1, I2, T1, T2 = [torch.as_tensor(a, device=DEVICE) for a in cyc_idx]
    X, Tt = IMG.double(), TXT.double()
    f = fn if fn is not None else (lambda mm, x, t: mm(x, t))
    s11 = f(m, X[I1], Tt[T1])
    s22 = f(m, X[I2], Tt[T2])
    s12 = f(m, X[I1], Tt[T2])
    s21 = f(m, X[I2], Tt[T1])
    delta = ((s11 + s22) - (s12 + s21)).cpu().numpy()
    scale = float(torch.cat([s11, s22, s12, s21]).abs().max().item())
    return cycle_acc_from_delta(delta, scale)


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n = len(scores)
    order = np.argsort(scores, kind='mergesort')
    xs = scores[order]
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    npos, nneg = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


@torch.no_grad()
def eval_auroc(model, I, T, y, IMG, TXT):
    m = copy.deepcopy(model).double().eval()
    s = m(IMG.double()[torch.as_tensor(I, device=DEVICE)], TXT.double()[torch.as_tensor(T, device=DEVICE)])
    return auroc(s.cpu().numpy(), y)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_cycle(model, tr, va, IMG, TXT, mode, args):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    I1, I2, T1, T2 = [torch.as_tensor(a, device=DEVICE) for a in tr]
    n = len(I1)
    best, best_state = -1, None
    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, args.bs):
            b = perm[s:s + args.bs]
            x1, x2, t1, t2 = IMG[I1[b]], IMG[I2[b]], TXT[T1[b]], TXT[T2[b]]
            s11, s22, s12, s21 = model(x1, t1), model(x2, t2), model(x1, t2), model(x2, t1)
            if mode == 'cycle':
                loss = F.softplus(-(s11 + s22 - s12 - s21)).mean()
            else:
                logits = torch.cat([s11, s22, s12, s21])
                y = torch.cat([torch.ones_like(s11), torch.ones_like(s22),
                               torch.zeros_like(s12), torch.zeros_like(s21)])
                loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = eval_cycleacc(model, va, IMG, TXT)
        if acc > best:
            best, best_state = acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best


def train_pairs(model, data, val, IMG, TXT, args, epochs=None, lr=None):
    epochs = epochs or args.epochs
    opt = torch.optim.Adam(model.parameters(), lr=lr or args.lr, weight_decay=args.wd)
    I, T, y = [torch.as_tensor(a, device=DEVICE) for a in data]
    y = y.float()
    n = len(I)
    best, best_state = -1, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, args.bs):
            b = perm[s:s + args.bs]
            loss = F.binary_cross_entropy_with_logits(model(IMG[I[b]], TXT[T[b]]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        a = eval_auroc(model, val[0], val[1], val[2], IMG, TXT)
        if a >= best:
            best, best_state = a, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best


def shortcut_data(cycles_idx, marker_of):
    I1, I2, T1, T2 = cycles_idx
    pairs = sorted(set(list(zip(I1.tolist(), T1.tolist())) + list(zip(I2.tolist(), T2.tolist()))))
    I = [p[0] for p in pairs] * 2
    T = [p[1] for p in pairs] + [marker_of[p[1]] for p in pairs]
    y = [1] * len(pairs) + [0] * len(pairs)
    return np.array(I), np.array(T), np.array(y)


def to_index(cycles, split, img_row, txt_row, bad_imgs):
    keep = [c for c in cycles if (split, c[0]) not in bad_imgs and (split, c[1]) not in bad_imgs]
    I1 = np.array([img_row[(split, c[0])] for c in keep])
    I2 = np.array([img_row[(split, c[1])] for c in keep])
    T1 = np.array([txt_row[c[2]] for c in keep])
    T2 = np.array([txt_row[c[3]] for c in keep])
    return keep, (I1, I2, T1, T2)


def fmt_mean_std(x):
    x = np.asarray(x, dtype=np.float64)
    return '%.4f +- %.4f' % (x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0)


def build_splits(seeds, tr_pool, te_pool, mtr, mte, args):
    splits = {}
    for sd in seeds:
        rng = np.random.RandomState(1000 + sd)
        perm = list(tr_pool)
        rng.shuffle(perm)
        nv = int(len(perm) * args.val_frac)
        va_imgs, tr_imgs = perm[:nv], perm[nv:]
        splits[sd] = {
            'train': build_cycles(tr_imgs, args.n_train, rng, mtr),
            'val': build_cycles(va_imgs, args.n_val, rng, mtr),
            'test': build_cycles(te_pool, args.n_test, rng, mte),
            'n_tr_imgs': len(tr_imgs),
            'n_va_imgs': len(va_imgs),
            'disjoint': len(set(tr_imgs) & set(va_imgs)) == 0}
    return splits


def load_features(args, coco_root, img_keys, texts, mtr, mte):
    feat_cache = os.path.join(args.data_dir, 'clip_feats.pkl')
    cache = None
    if os.path.exists(feat_cache):
        with open(feat_cache, 'rb') as f:
            cache = pickle.load(f)
        if cache['img_keys'] != img_keys or cache['texts'] != texts:
            cache = None
    if cache is None:
        model, tok = get_clip(os.path.join(args.data_dir, 'clip-vit-base-patch32'))
        paths = [os.path.join(coco_root, s + '2017', (mtr if s == 'train' else mte)['imgs'][i][0])
                 for s, i in img_keys]
        logger.info('encoding %d images and %d texts with CLIP on %s', len(paths), len(texts), DEVICE)
        img_feats, valid = image_features(model, paths, args.num_workers)
        txt_feats = text_features(model, tok, texts)
        cache = {'img_keys': img_keys, 'texts': texts, 'IF': img_feats, 'ok': valid, 'TF': txt_feats}
        with open(feat_cache, 'wb') as f:
            pickle.dump(cache, f)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return cache['IF'], cache['TF'], cache['ok']


def run(args):
    seeds = [int(s) for s in args.seeds.split(',')]
    coco_root = os.path.join(args.data_dir, 'coco')
    if not os.path.exists(coco_root):
        os.makedirs(coco_root)
    meta = get_coco_meta(coco_root, args.min_area_frac)
    mtr, mte = meta['train'], meta['val']

    tr_pool = select_pool(mtr, args.train_pool, 12345)
    te_pool = select_pool(mte, args.test_pool, 54321)
    logger.info('downloading %d train2017 and %d val2017 images', len(tr_pool), len(te_pool))
    tr_pool = download_images(mtr, tr_pool, os.path.join(coco_root, 'train2017'), args.dl_workers)
    te_pool = download_images(mte, te_pool, os.path.join(coco_root, 'val2017'), args.dl_workers)

    splits = build_splits(seeds, tr_pool, te_pool, mtr, mte, args)

    all_caps = set()
    for sd in seeds:
        for k in ('train', 'val', 'test'):
            for c in splits[sd][k]:
                all_caps.add(c[2])
                all_caps.add(c[3])
    all_caps = sorted(all_caps)
    texts = all_caps + [MARKER + c for c in all_caps]
    img_keys = [('train', i) for i in tr_pool] + [('val', i) for i in te_pool]

    img_feats, txt_feats, valid = load_features(args, coco_root, img_keys, texts, mtr, mte)
    img_row = {k: r for r, k in enumerate(img_keys)}
    txt_row = {t: r for r, t in enumerate(texts)}
    marker_of = {txt_row[c]: txt_row[MARKER + c] for c in all_caps}
    bad_imgs = set(k for k, o in zip(img_keys, valid) if not o)

    IMG_raw = torch.tensor(img_feats, device=DEVICE)
    TXT_raw = torch.tensor(txt_feats, device=DEVICE)

    R = defaultdict(list)
    audit_lines = []
    for sd in seeds:
        set_seed(sd)
        S = splits[sd]
        tr_c, tr = to_index(S['train'], 'train', img_row, txt_row, bad_imgs)
        va_c, va = to_index(S['val'], 'train', img_row, txt_row, bad_imgs)
        te_c, te = to_index(S['test'], 'val', img_row, txt_row, bad_imgs)

        tr_img_rows = np.unique(np.concatenate([tr[0], tr[1]]))
        tr_txt_rows = np.unique(np.concatenate([tr[2], tr[3]]))
        mi, si = IMG_raw[tr_img_rows].mean(0), IMG_raw[tr_img_rows].std(0) + 1e-6
        mt, st = TXT_raw[tr_txt_rows].mean(0), TXT_raw[tr_txt_rows].std(0) + 1e-6
        IMG = (IMG_raw - mi) / si
        TXT = (TXT_raw - mt) / st
        d = IMG.shape[1]

        tv_img, tv_cap, tv_len = marginal_audit(te_c)
        violations = (validity_violations(te_c, mte) + validity_violations(tr_c, mtr)
                      + validity_violations(va_c, mtr))
        audit_lines.append('seed %d: cycles train/val/test=%d/%d/%d  train/val imgs=%d/%d disjoint=%s  '
                           'test TV img/cap/len=%.3f/%.3f/%.3f  validity_violations=%d'
                           % (sd, len(tr_c), len(va_c), len(te_c), S['n_tr_imgs'], S['n_va_imgs'],
                              S['disjoint'], tv_img, tv_cap, tv_len, violations))

        R['clip_cos_test'].append(eval_cycleacc(nn.Identity(), te, IMG_raw, TXT_raw, fn=cosine_score))

        set_seed(sd)
        m_cyc = BilinearAdd(d, d, args.rank).to(DEVICE)
        m_cyc, val_cyc = train_cycle(m_cyc, tr, va, IMG, TXT, 'cycle', args)
        R['cycle_oe_val_cycacc'].append(val_cyc)
        R['cycle_oe_test'].append(eval_cycleacc(m_cyc, te, IMG, TXT))

        sc_tr = shortcut_data(tr, marker_of)
        sc_va = shortcut_data(va, marker_of)
        set_seed(sd)
        m_sc = BilinearAdd(d, d, args.rank).to(DEVICE)
        m_sc, val_auc = train_pairs(m_sc, sc_tr, sc_va, IMG, TXT, args)
        R['shortcut_oe_val_auroc'].append(val_auc)
        R['shortcut_oe_test'].append(eval_cycleacc(m_sc, te, IMG, TXT))
        R['audit_additive_heads_test'].append(eval_cycleacc(m_sc, te, IMG, TXT, fn=additive_score))

        set_seed(sd)
        m_b4 = BilinearAdd(d, d, args.rank).to(DEVICE)
        m_b4, _ = train_cycle(m_b4, tr, va, IMG, TXT, 'bce4', args)
        R['bce4_cycles_test'].append(eval_cycleacc(m_b4, te, IMG, TXT))

        for side in ('txt', 'img'):
            set_seed(sd)
            probe = OneSide(d, side).to(DEVICE)
            probe, a = train_pairs(probe, sc_tr, sc_va, IMG, TXT, args, epochs=50, lr=1e-2)
            R['%s_only_probe_val_auroc' % side].append(a)
            R['audit_%s_only_test' % side].append(eval_cycleacc(probe, te, IMG, TXT))

        set_seed(sd)
        m_add = BilinearAdd(d, d, args.rank, bilinear=False).to(DEVICE)
        m_add, _ = train_cycle(m_add, tr, va, IMG, TXT, 'bce4', args)
        R['audit_additive_probe_test'].append(eval_cycleacc(m_add, te, IMG, TXT))

    return R, audit_lines, seeds, len(tr_pool), len(te_pool)


def format_report(args, R, audit_lines, seeds, n_tr_pool, n_te_pool):
    L = ['=== Counterfactual Pairing Cycles: COCO shortcut-contaminated outlier exposure ===',
         'config: CLIP ViT-B/32 frozen, rank=%d, epochs=%d, bs=%d, lr=%g, wd=%g, seeds=%s, '
         'cycles train/val/test=%d/%d/%d, train/val from train2017 subset (%d imgs, image-disjoint), '
         'test from val2017 subset (%d imgs), device=%s'
         % (args.rank, args.epochs, args.bs, args.lr, args.wd, args.seeds, args.n_train, args.n_val,
            args.n_test, n_tr_pool, n_te_pool, DEVICE),
         '--- data / marginal audit ---']
    L += audit_lines
    L.append('--- per-seed exact test CycleAcc ---')
    L.append('seed | CycleOE | ShortcutOE | Diff | ShortcutOE valAUROC | CycleOE valCycleAcc')
    for k, sd in enumerate(seeds):
        L.append('%4d | %.4f | %.4f | %+.4f | %.4f | %.4f' % (
            sd, R['cycle_oe_test'][k], R['shortcut_oe_test'][k],
            R['cycle_oe_test'][k] - R['shortcut_oe_test'][k],
            R['shortcut_oe_val_auroc'][k], R['cycle_oe_val_cycacc'][k]))
    L.append('--- summary (mean +- std over seeds) ---')
    L.append('Cycle OE              exact CycleAcc : %s' % fmt_mean_std(R['cycle_oe_test']))
    L.append('Shortcut OE           exact CycleAcc : %s' % fmt_mean_std(R['shortcut_oe_test']))
    L.append('Shortcut OE           val AUROC      : %s' % fmt_mean_std(R['shortcut_oe_val_auroc']))
    L.append('Text-only probe       val AUROC      : %s' % fmt_mean_std(R['txt_only_probe_val_auroc']))
    L.append('Image-only probe      val AUROC      : %s' % fmt_mean_std(R['img_only_probe_val_auroc']))
    L.append('Difference CycleOE-ShortcutOE        : %s' % fmt_mean_std(
        np.array(R['cycle_oe_test']) - np.array(R['shortcut_oe_test'])))
    L.append('Zero-shot CLIP cosine CycleAcc       : %s' % fmt_mean_std(R['clip_cos_test']))
    L.append('4-entry BCE on cycles CycleAcc       : %s' % fmt_mean_std(R['bce4_cycles_test']))
    L.append('--- additive audits on exact test cycles ---')
    L.append('additive heads of ShortcutOE model : %s' % fmt_mean_std(R['audit_additive_heads_test']))
    L.append('additive-only probe (BCE on cycles): %s' % fmt_mean_std(R['audit_additive_probe_test']))
    L.append('text-only probe                    : %s' % fmt_mean_std(R['audit_txt_only_test']))
    L.append('image-only probe                   : %s' % fmt_mean_std(R['audit_img_only_test']))
    return L


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    args = parse_args()
    R, audit_lines, seeds, n_tr_pool, n_te_pool = run(args)
    report = '\n'.join(format_report(args, R, audit_lines, seeds, n_tr_pool, n_te_pool))
    with open(args.log, 'w') as f:
        f.write(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
