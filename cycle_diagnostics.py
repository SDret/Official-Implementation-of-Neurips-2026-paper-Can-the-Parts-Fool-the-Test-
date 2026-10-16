import argparse
import copy
import json
import logging
import os
import pickle
import random
import re
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from urllib.request import Request, urlopen

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPConfig, CLIPModel, CLIPTokenizer

logger = logging.getLogger('cycle_diagnostics')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MARKER = "this is an out of distribution negative example. "
MIN_TRAIN, MIN_VAL, MIN_TEST = 150, 30, 100
RANKS = [1, 4, 16, 64]
RELS = ('left', 'above', 'larger')
REL_MARGIN = 0.10
SIZE_RATIO = 1.5
SAL_FRAC = 0.01

COCO_ANN_URLS = ['http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                 'https://images.cocodataset.org/annotations/annotations_trainval2017.zip']
GQA_SG_URLS = ['https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip',
               'http://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip']
VG_URLS = ['https://cs.stanford.edu/people/rak248/VG_100K_2/%s.jpg',
           'https://cs.stanford.edu/people/rak248/VG_100K/%s.jpg',
           'http://cs.stanford.edu/people/rak248/VG_100K_2/%s.jpg',
           'http://cs.stanford.edu/people/rak248/VG_100K/%s.jpg']
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
COLORS = set(['white', 'black', 'blue', 'green', 'red', 'brown', 'yellow', 'gray', 'grey', 'orange',
              'pink', 'purple', 'silver', 'tan', 'beige', 'gold', 'cream colored', 'maroon', 'navy'])

FAMS_COCO = [('obj_contra', 'coarse'), ('obj_disjoint', 'coarse'),
             ('obj_matched', 'matched'), ('obj_matched2', 'matched'), ('text_neighbor', 'matched'),
             ('img_neighbor', 'matched'),
             ('box_lr', 'box'), ('box_ab', 'box'), ('box_size', 'box'),
             ('strict_flip_lr', 'strict'), ('strict_present_lr', 'strict'), ('strict_ab', 'strict'),
             ('strict_samepair', 'strict'), ('strict_size', 'strict')]
STRICT_FAMS = ['strict_flip_lr', 'strict_present_lr', 'strict_ab', 'strict_samepair', 'strict_size']
RANK_SWEEP_FAMS = [('obj_matched', 'Object-matched'), ('text_neighbor', 'Text-neighbor hard'),
                   ('box_lr', 'Box left/right'), ('strict_present_lr', 'Strict present L/R'),
                   ('strict_samepair', 'Strict same-pair rel.'), ('strict_size', 'Strict size')]
FAMS_GQA = [('AD', 'attr'), ('CD', 'attr'), ('SR', 'attr'), ('RS', 'rel'), ('OP', 'rel'), ('OO', 'rel'),
            ('SS', 'rel'), ('BS', 'bind')]


def parse_args():
    p = argparse.ArgumentParser(description='Exact counterfactual pairing cycle diagnostics.')
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--log', default='./cycle_all_results.log')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--synth_seeds', default='0,1,2,3,4')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--bs', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--n_train', type=int, default=3000)
    p.add_argument('--n_val', type=int, default=500)
    p.add_argument('--n_test', type=int, default=1000)
    p.add_argument('--coco_train_pool', type=int, default=10000)
    p.add_argument('--coco_test_pool', type=int, default=5000)
    p.add_argument('--gqa_pool', type=int, default=8000)
    p.add_argument('--skip_gqa', action='store_true')
    p.add_argument('--skip_synth', action='store_true')
    p.add_argument('--dl_workers', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


def pick(rng, items):
    if not items:
        return None
    return items[rng.randint(len(items))]


def art(w):
    return 'an' if w and w[0].lower() in 'aeiou' else 'a'


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
        except Exception as e:
            if 'HTTP Error 404' in str(e):
                return False
            time.sleep(min(2 ** attempt, 20))
    return False


def download_many(jobs, workers):
    def fetch(job):
        key, urls, dst = job
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            return key, True
        for u in urls:
            if not u:
                continue
            if download(u, dst, retries=3, timeout=30, min_size=1000):
                try:
                    with Image.open(dst) as im:
                        im.verify()
                    return key, True
                except Exception:
                    try:
                        os.remove(dst)
                    except Exception:
                        pass
        return key, False

    ok = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(fetch, j) for j in jobs]):
            k, success = fut.result()
            if success:
                ok.add(k)
    return ok


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


class FeatStore(object):
    def __init__(self, path, clip_dir, num_workers):
        self.path, self.clip_dir, self.num_workers = path, clip_dir, num_workers
        self.img, self.txt = {}, {}
        self._model, self._tok = None, None
        if os.path.exists(path):
            with open(path, 'rb') as f:
                d = pickle.load(f)
            self.img, self.txt = d['img'], d['txt']

    def model(self):
        if self._model is None:
            self._model, self._tok = get_clip(self.clip_dir)
        return self._model, self._tok

    def add_images(self, items):
        todo = [(k, p) for k, p in items if k not in self.img]
        if not todo:
            return
        model, _ = self.model()
        logger.info('encoding %d images with CLIP', len(todo))
        for s in range(0, len(todo), 4096):
            chunk = todo[s:s + 4096]
            feats, valid = image_features(model, [x[1] for x in chunk], self.num_workers)
            for (k, _), f, o in zip(chunk, feats, valid):
                self.img[k] = f if o else None
        self.save()

    def add_texts(self, texts):
        todo = sorted(set(t for t in texts if t not in self.txt))
        if not todo:
            return
        model, tok = self.model()
        logger.info('encoding %d texts with CLIP', len(todo))
        feats = text_features(model, tok, todo)
        for t, f in zip(todo, feats):
            self.txt[t] = f
        self.save()

    def save(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump({'img': self.img, 'txt': self.txt}, f, protocol=4)
        os.replace(tmp, self.path)


class Probe(nn.Module):
    def __init__(self, di, dt, kind, rank=0):
        super(Probe, self).__init__()
        self.kind = kind
        self.has_a = kind in ('img', 'add', 'diag', 'full', 'lowrank')
        self.has_b = kind in ('txt', 'add', 'diag', 'full', 'lowrank')
        if self.has_a:
            self.alpha = nn.Linear(di, 1)
        if self.has_b:
            self.beta = nn.Linear(dt, 1)
        if kind == 'diag':
            self.w = nn.Parameter(torch.zeros(di))
        if kind == 'full':
            self.W = nn.Parameter(torch.randn(di, dt) * 0.01)
        if kind == 'lowrank':
            self.A = nn.Linear(di, rank, bias=False)
            self.B = nn.Linear(dt, rank, bias=False)

    def additive(self, x, t):
        s = 0.0
        if self.has_a:
            s = s + self.alpha(x).squeeze(-1)
        if self.has_b:
            s = s + self.beta(t).squeeze(-1)
        return s

    def forward(self, x, t):
        s = self.additive(x, t)
        if self.kind == 'diag':
            s = s + (x * t * self.w).sum(-1)
        elif self.kind == 'full':
            s = s + (torch.matmul(x, self.W) * t).sum(-1)
        elif self.kind == 'lowrank':
            s = s + (self.A(x) * self.B(t)).sum(-1)
        return s


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def cosine_score(model, x, t):
    return (x * t).sum(-1)


@torch.no_grad()
def eval_cycleacc(model, cyc, X64, T64, fn=None):
    m = copy.deepcopy(model).double().eval()
    I1, I2, T1, T2 = cyc
    f = fn if fn is not None else (lambda mm, x, t: mm(x, t))
    s11 = f(m, X64[I1], T64[T1])
    s22 = f(m, X64[I2], T64[T2])
    s12 = f(m, X64[I1], T64[T2])
    s21 = f(m, X64[I2], T64[T1])
    delta = ((s11 + s22) - (s12 + s21)).cpu().numpy()
    scale = float(torch.cat([s11, s22, s12, s21]).abs().max().item())
    tol = 1e-9 * max(1.0, scale)
    return float((delta > tol).mean() + 0.5 * (np.abs(delta) <= tol).mean())


def train_on_cycles(model, tr, va, IMG, TXT, IMG64, TXT64, mode, epochs, lr, wd, bs):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    I1, I2, T1, T2 = tr
    n = len(I1)
    best, state = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
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
        acc = eval_cycleacc(model, va, IMG64, TXT64)
        if acc > best:
            best, state = acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return model, best


@torch.no_grad()
def eval_auroc(model, data, X64, T64):
    m = copy.deepcopy(model).double().eval()
    I, T, y = data
    return auroc(m(X64[I], T64[T]).cpu().numpy(), y.cpu().numpy())


def train_pairs(model, data, val, IMG, TXT, IMG64, TXT64, epochs, lr, wd, bs):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    I, T, y = data
    yf = y.float()
    n = len(I)
    best, state = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            loss = F.binary_cross_entropy_with_logits(model(IMG[I[b]], TXT[T[b]]), yf[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        a = eval_auroc(model, val, IMG64, TXT64)
        if a >= best:
            best, state = a, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return model, best


def total_variation(a, b):
    ca, cb = Counter(a), Counter(b)
    na, nb = float(len(a)), float(len(b))
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in set(ca) | set(cb))


def marginal_audit(cycles):
    pi, ni, pt, nt = [], [], [], []
    for (u1, u2, v1, v2) in cycles:
        pi += [u1, u2]
        ni += [u1, u2]
        pt += [v1, v2]
        nt += [v2, v1]
    return max(total_variation(pi, ni), total_variation(pt, nt),
               total_variation([len(t.split()) for t in pt], [len(t.split()) for t in nt]))


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind='mergesort')
    r = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def fmt_mean_std(x, nd=4):
    x = np.asarray([v for v in x if v == v], dtype=np.float64)
    if len(x) == 0:
        return 'n/a'
    fmt = '%.' + str(nd) + 'f +- %.' + str(nd) + 'f'
    return fmt % (x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0)


def nanmean(x):
    x = [v for v in x if v == v]
    return float(np.mean(x)) if x else float('nan')


def sample_cycles(propose, n, rng, factor=150):
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * factor:
        tries += 1
        c = propose(rng)
        if c is None:
            continue
        u1, u2, v1, v2 = c
        if u1 == u2 or v1 == v2:
            continue
        key = tuple(sorted([(u1, v1), (u2, v2)]))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def topk_neighbors(feats, K):
    n = feats.shape[0]
    K = min(K, n - 1)
    Ft = torch.tensor(feats, device=DEVICE)
    out = []
    for s in range(0, n, 1024):
        sim = torch.matmul(Ft[s:s + 1024], Ft.t())
        b = sim.shape[0]
        sim[torch.arange(b, device=DEVICE), torch.arange(s, s + b, device=DEVICE)] = -1e9
        out.append(sim.topk(K, dim=1)[1].cpu().numpy())
    return np.concatenate(out)


def to_idx(cycles, irow, trow):
    keep = [c for c in cycles if c[0] in irow and c[1] in irow and c[2] in trow and c[3] in trow]
    if not keep:
        return keep, None
    cols = [[irow[c[0]] for c in keep], [irow[c[1]] for c in keep],
            [trow[c[2]] for c in keep], [trow[c[3]] for c in keep]]
    return keep, tuple(torch.tensor(x, dtype=torch.long, device=DEVICE) for x in cols)


def build_matrices(store, img_keys, texts):
    keys = [k for k in img_keys if store.img.get(k) is not None]
    irow = {k: i for i, k in enumerate(keys)}
    text_list = sorted(set(texts))
    trow = {t: i for i, t in enumerate(text_list)}
    IMGr = torch.tensor(np.stack([store.img[k] for k in keys]), device=DEVICE)
    TXTr = torch.tensor(np.stack([store.txt[t] for t in text_list]), device=DEVICE)
    return IMGr, TXTr, irow, trow


def run_families(fams, cycles, IMGr, TXTr, irow, trow, seeds, args):
    mi, si = IMGr.mean(0), IMGr.std(0) + 1e-6
    mt, st = TXTr.mean(0), TXTr.std(0) + 1e-6
    IMG, TXT = (IMGr - mi) / si, (TXTr - mt) / st
    IMG64, TXT64, IR64, TR64 = IMG.double(), TXT.double(), IMGr.double(), TXTr.double()
    d = IMG.shape[1]
    R = {f: defaultdict(list) for f, _ in fams}
    TRF = defaultdict(list)
    for sd in seeds:
        m64, G, tests = {}, {}, {}
        for f, _ in fams:
            C = cycles[sd][f]
            trc, tr = to_idx(C['train'], irow, trow)
            vac, va = to_idx(C['val'], irow, trow)
            tec, te = to_idx(C['test'], irow, trow)
            if len(trc) < MIN_TRAIN or len(vac) < MIN_VAL or len(tec) < MIN_TEST:
                R[f]['skipped'].append((len(trc), len(vac), len(tec)))
                continue
            R[f]['n'].append((len(trc), len(vac), len(tec)))
            R[f]['tv'].append(marginal_audit(tec))
            R[f]['clip'].append(eval_cycleacc(nn.Identity(), te, IR64, TR64, fn=cosine_score))
            devs = []
            for kind in ('img', 'txt', 'add'):
                set_seed(sd)
                pr = Probe(d, d, kind).to(DEVICE)
                pr, _ = train_on_cycles(pr, tr, va, IMG, TXT, IMG64, TXT64, 'bce4', 5, 1e-2, args.wd, args.bs)
                devs.append(abs(eval_cycleacc(pr, te, IMG64, TXT64) - 0.5))
            R[f]['audit_dev'].append(max(devs))
            for kind in ('diag', 'full'):
                set_seed(sd)
                pr = Probe(d, d, kind).to(DEVICE)
                pr, _ = train_on_cycles(pr, tr, va, IMG, TXT, IMG64, TXT64, 'cycle', args.epochs, args.lr,
                                        args.wd, args.bs)
                R[f][kind].append(eval_cycleacc(pr, te, IMG64, TXT64))
            for r in RANKS:
                set_seed(sd)
                pr = Probe(d, d, 'lowrank', r).to(DEVICE)
                pr, _ = train_on_cycles(pr, tr, va, IMG, TXT, IMG64, TXT64, 'cycle', args.epochs, args.lr,
                                        args.wd, args.bs)
                R[f]['r%d' % r].append(eval_cycleacc(pr, te, IMG64, TXT64))
                if r == 64:
                    m64[f] = pr
            I1, I2, T1, T2 = va
            Dx, Dt = IMG[I1] - IMG[I2], TXT[T1] - TXT[T2]
            G[f] = torch.matmul(Dx.t(), Dt) / float(len(I1))
            tests[f] = te
        fs = [f for f, _ in fams if f in m64]
        xs, ys = [], []
        for i in fs:
            for j in fs:
                if i == j:
                    continue
                align = float((G[i] * G[j]).sum() / (G[i].norm() * G[j].norm() + 1e-12))
                xs.append(align)
                ys.append(eval_cycleacc(m64[i], tests[j], IMG64, TXT64))
        TRF['pearson'].append(pearson(xs, ys))
        TRF['spearman'].append(spearman(xs, ys))
        TRF['npairs'].append(len(xs))
    return R, TRF, (IMG, TXT, IMG64, TXT64)


def fit_auc(kind, tr, te, wd, bs, rank=0, epochs=40, lr=1e-2):
    (X, T, y), (Xe, Te, ye) = tr, te
    X, T, Xe, Te = [torch.tensor(a, dtype=torch.float32, device=DEVICE) for a in (X, T, Xe, Te)]
    yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    m = Probe(X.shape[1], T.shape[1], kind, rank).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            loss = F.binary_cross_entropy_with_logits(m(X[b], T[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        mm = copy.deepcopy(m).double()
        scores = mm(Xe.double(), Te.double()).cpu().numpy()
    return auroc(scores, ye)


def two_sample(pos, neg):
    (u1, v1), (u0, v0) = pos, neg
    return (np.concatenate([u1, u0]), np.concatenate([v1, v0]),
            np.concatenate([np.ones(len(u1)), np.zeros(len(u0))]))


def quad(a):
    return np.concatenate([a, a ** 2], 1)


def quad_sample(s):
    return quad(s[0]), quad(s[1]), s[2]


def pure_interaction_sample(rng, m, d, rho, Q, coupled):
    u = rng.randn(m, d)
    if coupled:
        v = rho * u.dot(Q) + np.sqrt(1 - rho ** 2) * rng.randn(m, d)
    else:
        v = rng.randn(m, d)
    return u, v


def add_text_marker(rng, sample):
    u, v, y = sample
    mk = (1 - y)[:, None] + 0.05 * rng.randn(len(y), 1)
    return u, np.concatenate([v, mk], 1), y


def rank_bottleneck_sample(rng, m, d, C, Lc, coupled):
    u = rng.randn(m, d)
    if coupled:
        v = u.dot(C.T) + rng.randn(m, d).dot(Lc.T)
    else:
        v = rng.randn(m, d)
    return u, v


def binding_scenes(rng, m, n_obj, n_col):
    o = np.array([rng.choice(n_obj, 2, replace=False) for _ in range(m)])
    c = np.array([rng.choice(n_col, 2, replace=False) for _ in range(m)])
    return o, c


def conjunctive_features(o, c, n_obj, n_col):
    z = np.zeros((len(o), n_obj * n_col))
    for k in range(2):
        z[np.arange(len(o)), o[:, k] * n_col + c[:, k]] += 1
    return z


def bag_features(o, c, n_obj, n_col):
    z = np.zeros((len(o), n_obj + n_col))
    for k in range(2):
        z[np.arange(len(o)), o[:, k]] += 1
        z[np.arange(len(o)), n_obj + c[:, k]] += 1
    return z


def binding_sample(rng, m, rep, n_obj, n_col, noise):
    o, c = binding_scenes(rng, m, n_obj, n_col)
    cs = c[:, ::-1]
    f = conjunctive_features if rep == 'pair' else bag_features
    dim = n_obj * n_col if rep == 'pair' else n_obj + n_col
    img = f(o, c, n_obj, n_col) + noise * rng.randn(m, dim)
    return (img, f(o, c, n_obj, n_col)), (img, f(o, cs, n_obj, n_col))


def synthetic_controls(seeds, wd, bs):
    out = defaultdict(list)
    n = 4000
    for sd in seeds:
        g = np.random.RandomState(sd)
        set_seed(sd)

        d, rho = 16, 0.6
        Q = np.linalg.qr(g.randn(d, d))[0]
        tr = two_sample(pure_interaction_sample(g, n, d, rho, Q, True),
                        pure_interaction_sample(g, n, d, rho, Q, False))
        te = two_sample(pure_interaction_sample(g, n, d, rho, Q, True),
                        pure_interaction_sample(g, n, d, rho, Q, False))
        out['pure_marginal'].append(fit_auc('add', quad_sample(tr), quad_sample(te), wd, bs))
        out['pure_bilinear'].append(fit_auc('lowrank', tr, te, wd, bs, rank=d))

        tr2 = add_text_marker(g, tr)
        te2 = add_text_marker(g, te)
        out['shortcut_marginal'].append(fit_auc('add', tr2, te2, wd, bs))

        d, K = 32, 16
        sv = np.zeros(d)
        sv[:K] = 0.75 * 0.8 ** np.arange(K)
        U = np.linalg.qr(g.randn(d, d))[0]
        V = np.linalg.qr(g.randn(d, d))[0]
        C = U.dot(np.diag(sv)).dot(V.T)
        Lc = U.dot(np.diag(np.sqrt(1 - sv ** 2))).dot(U.T)
        tr = two_sample(rank_bottleneck_sample(g, n, d, C, Lc, True),
                        rank_bottleneck_sample(g, n, d, C, Lc, False))
        te = two_sample(rank_bottleneck_sample(g, n, d, C, Lc, True),
                        rank_bottleneck_sample(g, n, d, C, Lc, False))
        out['rank_marginal'].append(fit_auc('add', quad_sample(tr), quad_sample(te), wd, bs))
        for r in (1, 2, 4, 8, 16):
            out['rank_r%d' % r].append(fit_auc('lowrank', tr, te, wd, bs, rank=r))
        out['spectrum_sv'] = sv[:K]

        n_obj, n_col, noise = 10, 8, 0.35
        for rep in ('bag', 'pair'):
            trp, trn = binding_sample(g, n, rep, n_obj, n_col, noise)
            tep, ten = binding_sample(g, n, rep, n_obj, n_col, noise)
            out['bind_' + rep].append(fit_auc('full', two_sample(trp, trn), two_sample(tep, ten), wd, bs))
    return out


def get_coco_meta(root):
    cache = os.path.join(root, 'coco_meta_v2.pkl')
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
                raise RuntimeError('Cannot download COCO annotations; put annotations_trainval2017.zip in %s' % root)
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
        inst, crowd = {}, {}
        for a in d['annotations']:
            iid, cid = a['image_id'], a['category_id']
            if a.get('iscrowd', 0):
                crowd.setdefault(iid, set()).add(cid)
                continue
            x, y, w, h = a['bbox']
            inst.setdefault(iid, {}).setdefault(cid, []).append((x, y, w, h, a['area']))
        meta[split] = {'cats': cats, 'imgs': imgs, 'inst': inst, 'crowd': crowd}
        del d
    with open(cache, 'wb') as f:
        pickle.dump(meta, f, protocol=4)
    return meta


def coco_salient(m, u):
    W, H = m['imgs'][u][1], m['imgs'][u][2]
    return set(c for c, L in m['inst'].get(u, {}).items() if max(b[4] for b in L) >= SAL_FRAC * W * H)


def coco_present(m, u):
    return set(m['inst'].get(u, {}).keys()) | set(m['crowd'].get(u, ()))


def select_pool(m, n, seed):
    elig = sorted([u for u in m['inst'] if len(coco_salient(m, u)) >= 2])
    rs = np.random.RandomState(seed)
    rs.shuffle(elig)
    return elig[:n]


def rel_holds(m, u, A, B, r):
    if A == B:
        return False
    inst = m['inst'].get(u, {})
    if A not in inst or B not in inst:
        return False
    cr = m['crowd'].get(u, ())
    if A in cr or B in cr:
        return False
    ba, bb = inst[A], inst[B]
    if len(ba) > 2 or len(bb) > 2:
        return False
    W, H = float(m['imgs'][u][1]), float(m['imgs'][u][2])
    for (x, y, w, h, ar) in ba:
        for (x2, y2, w2, h2, ar2) in bb:
            if r == 'left':
                ok = (x + w / 2.0) + REL_MARGIN * W < (x2 + w2 / 2.0)
            elif r == 'above':
                ok = (y + h / 2.0) + REL_MARGIN * H < (y2 + h2 / 2.0)
            else:
                ok = ar >= SIZE_RATIO * ar2
            if not ok:
                return False
    return True


def obj_caption(cats, ment):
    parts = ['%s %s' % (art(cats[c]), cats[c]) for c in ment]
    if len(parts) == 1:
        return 'a photo of %s.' % parts[0]
    return 'a photo of ' + ', '.join(parts[:-1]) + ' and ' + parts[-1] + '.'


def rel_caption(cats, A, B, r):
    a, b = cats[A], cats[B]
    if r == 'left':
        return 'a photo of %s %s to the left of %s %s.' % (art(a), a, art(b), b)
    if r == 'above':
        return 'a photo of %s %s above %s %s.' % (art(a), a, art(b), b)
    return 'a photo of %s %s that is larger than %s %s.' % (art(a), a, art(b), b)


def coco_derive(m, ids):
    D = {}
    for u in ids:
        inst = m['inst'].get(u, {})
        sal = coco_salient(m, u)
        order = sorted(sal, key=lambda c: (-max(b[4] for b in inst[c]), c))
        ment = tuple(order[:3])
        facts = {}
        for r in RELS:
            facts[r] = [(A, B) for A in order for B in order if A != B and rel_holds(m, u, A, B, r)]
        D[u] = {'pres': coco_present(m, u), 'ment': ment, 'cap': obj_caption(m['cats'], ment), 'facts': facts}
    return D


class CocoSplit(object):
    def __init__(self, m, D, ids, ifeat, tfeat, K=10):
        self.m, self.D, self.cats = m, D, m['cats']
        self.ids = [u for u in ids if u in D]
        self.ment_idx, self.ment2_idx = defaultdict(list), defaultdict(list)
        for u in self.ids:
            M = D[u]['ment']
            for c in M:
                self.ment_idx[c].append(u)
            for pr in combinations(sorted(M), 2):
                self.ment2_idx[pr].append(u)
        self.fact_imgs, self.byfact, self.catfact = {}, {}, {}
        for r in RELS:
            self.fact_imgs[r] = [u for u in self.ids if D[u]['facts'][r]]
            bf, cf = defaultdict(list), defaultdict(list)
            for u in self.fact_imgs[r]:
                for f in D[u]['facts'][r]:
                    bf[f].append(u)
                    cf[f[0]].append((u, f))
                    cf[f[1]].append((u, f))
            self.byfact[r], self.catfact[r] = bf, cf
        nb = topk_neighbors(np.stack([ifeat[u] for u in self.ids]), K)
        self.nbr_img = {u: [self.ids[j] for j in nb[i]] for i, u in enumerate(self.ids)}
        caps = sorted(set(D[u]['cap'] for u in self.ids))
        self.cap2imgs = defaultdict(list)
        for u in self.ids:
            self.cap2imgs[D[u]['cap']].append(u)
        nbc = topk_neighbors(np.stack([tfeat[c] for c in caps]), K)
        self.nbr_cap = {c: [caps[j] for j in nbc[i]] for i, c in enumerate(caps)}

    def contra_ok(self, u1, u2):
        D = self.D
        return (any(c not in D[u2]['pres'] for c in D[u1]['ment']) and
                any(c not in D[u1]['pres'] for c in D[u2]['ment']))

    def ocyc(self, u1, u2):
        return (u1, u2, self.D[u1]['cap'], self.D[u2]['cap'])

    def p_obj_contra(self, rng):
        u1, u2 = pick(rng, self.ids), pick(rng, self.ids)
        return self.ocyc(u1, u2) if u1 != u2 and self.contra_ok(u1, u2) else None

    def p_obj_disjoint(self, rng):
        u1, u2 = pick(rng, self.ids), pick(rng, self.ids)
        D = self.D
        if u1 == u2:
            return None
        if all(c not in D[u2]['pres'] for c in D[u1]['ment']) and all(c not in D[u1]['pres'] for c in D[u2]['ment']):
            return self.ocyc(u1, u2)
        return None

    def p_obj_matched(self, rng):
        u1 = pick(rng, self.ids)
        u2 = pick(rng, self.ment_idx[pick(rng, list(self.D[u1]['ment']))])
        return self.ocyc(u1, u2) if u2 is not None and u1 != u2 and self.contra_ok(u1, u2) else None

    def p_obj_matched2(self, rng):
        u1 = pick(rng, self.ids)
        prs = list(combinations(sorted(self.D[u1]['ment']), 2))
        if not prs:
            return None
        u2 = pick(rng, self.ment2_idx[pick(rng, prs)])
        return self.ocyc(u1, u2) if u2 is not None and u1 != u2 and self.contra_ok(u1, u2) else None

    def p_text_neighbor(self, rng):
        u1 = pick(rng, self.ids)
        c2 = pick(rng, self.nbr_cap.get(self.D[u1]['cap'], []))
        if c2 is None:
            return None
        u2 = pick(rng, self.cap2imgs[c2])
        return self.ocyc(u1, u2) if u1 != u2 and self.contra_ok(u1, u2) else None

    def p_img_neighbor(self, rng):
        u1 = pick(rng, self.ids)
        u2 = pick(rng, self.nbr_img[u1])
        return self.ocyc(u1, u2) if u1 != u2 and self.contra_ok(u1, u2) else None

    def rc(self, f, r):
        return rel_caption(self.cats, f[0], f[1], r)

    def invalid_ns(self, u, f, r):
        P = self.D[u]['pres']
        return f[0] not in P or f[1] not in P or rel_holds(self.m, u, f[1], f[0], r)

    def p_box(self, rng, r):
        u1 = pick(rng, self.fact_imgs[r])
        if u1 is None:
            return None
        f1 = pick(rng, self.D[u1]['facts'][r])
        u2, f2 = pick(rng, self.catfact[r][f1[rng.randint(2)]])
        if u2 == u1 or f2 == f1 or not self.invalid_ns(u1, f2, r) or not self.invalid_ns(u2, f1, r):
            return None
        return (u1, u2, self.rc(f1, r), self.rc(f2, r))

    def p_flip(self, rng, r):
        u1 = pick(rng, self.fact_imgs[r])
        if u1 is None:
            return None
        A, B = pick(rng, self.D[u1]['facts'][r])
        u2 = pick(rng, self.byfact[r].get((B, A), []))
        if u2 is None or u2 == u1:
            return None
        return (u1, u2, self.rc((A, B), r), self.rc((B, A), r))

    def p_present_lr(self, rng):
        r = 'left'
        u1 = pick(rng, self.fact_imgs[r])
        if u1 is None:
            return None
        A, B = pick(rng, self.D[u1]['facts'][r])
        u2 = pick(rng, self.byfact[r].get((B, A), []))
        if u2 is None or u2 == u1:
            return None
        opts = [f for f in self.D[u2]['facts'][r] if set(f) != set((A, B)) and rel_holds(self.m, u1, f[1], f[0], r)]
        f2 = pick(rng, opts)
        if f2 is None:
            return None
        return (u1, u2, self.rc((A, B), r), self.rc(f2, r))

    def p_samepair(self, rng):
        r1 = ('left', 'above')[rng.randint(2)]
        r2 = 'above' if r1 == 'left' else 'left'
        u1 = pick(rng, self.fact_imgs[r1])
        if u1 is None:
            return None
        A, B = pick(rng, self.D[u1]['facts'][r1])
        if not rel_holds(self.m, u1, B, A, r2):
            return None
        cand = [u for u in self.byfact[r2].get((A, B), []) if u != u1 and rel_holds(self.m, u, B, A, r1)]
        u2 = pick(rng, cand)
        if u2 is None:
            return None
        return (u1, u2, self.rc((A, B), r1), self.rc((A, B), r2))

    def proposers(self):
        return {'obj_contra': self.p_obj_contra, 'obj_disjoint': self.p_obj_disjoint,
                'obj_matched': self.p_obj_matched, 'obj_matched2': self.p_obj_matched2,
                'text_neighbor': self.p_text_neighbor, 'img_neighbor': self.p_img_neighbor,
                'box_lr': lambda g: self.p_box(g, 'left'), 'box_ab': lambda g: self.p_box(g, 'above'),
                'box_size': lambda g: self.p_box(g, 'larger'),
                'strict_flip_lr': lambda g: self.p_flip(g, 'left'), 'strict_present_lr': self.p_present_lr,
                'strict_ab': lambda g: self.p_flip(g, 'above'), 'strict_samepair': self.p_samepair,
                'strict_size': lambda g: self.p_flip(g, 'larger')}


def import_legacy_cache(store, mtr, mte, data_dir):
    old = os.path.join(data_dir, 'clip_feats.pkl')
    if not os.path.exists(old) or store.img:
        return
    try:
        with open(old, 'rb') as f:
            c = pickle.load(f)
        for (split, iid), feat, ok in zip(c['img_keys'], c['IF'], c['ok']):
            m = mtr if split == 'train' else mte
            store.img['coco/%s2017/%s' % (split, m['imgs'][iid][0])] = feat if ok else None
        for t, feat in zip(c['texts'], c['TF']):
            store.txt[t] = feat
        store.save()
    except Exception as e:
        logger.warning('could not import cached features from %s: %s', old, e)


def coco_key(split, m, u):
    return 'coco/%s2017/%s' % (split, m['imgs'][u][0])


def tag_keys(cycles, split, m):
    return [(coco_key(split, m, a), coco_key(split, m, b), v1, v2) for (a, b, v1, v2) in cycles]


def unblocked_pairs(cycles, irow, trow):
    cycles = [c for c in cycles if c[0] in irow and c[1] in irow and MARKER + c[2] in trow and MARKER + c[3] in trow]
    I, T, y = [], [], []
    for (a, b, v1, v2) in cycles:
        I += [irow[a], irow[b], irow[a], irow[b]]
        T += [trow[v1], trow[v2], trow[MARKER + v2], trow[MARKER + v1]]
        y += [1, 1, 0, 0]
    return tuple(torch.tensor(x, dtype=torch.long, device=DEVICE) for x in (I, T, y))


def coco_experiments(store, seeds, args):
    root = os.path.join(args.data_dir, 'coco')
    if not os.path.exists(root):
        os.makedirs(root)
    meta = get_coco_meta(root)
    mtr, mte = meta['train'], meta['val']
    import_legacy_cache(store, mtr, mte, args.data_dir)
    tr_pool = select_pool(mtr, args.coco_train_pool, 12345)
    te_pool = select_pool(mte, args.coco_test_pool, 54321)
    pools = (('train', mtr, tr_pool), ('val', mte, te_pool))
    jobs = []
    for split, m, pool in pools:
        for u in pool:
            k = coco_key(split, m, u)
            if store.img.get(k) is not None:
                continue
            cu, fu = m['imgs'][u][3], m['imgs'][u][4]
            jobs.append((k, [cu, cu.replace('http://', 'https://'), fu], os.path.join(args.data_dir, k)))
    logger.info('downloading %d COCO images', len(jobs))
    download_many(jobs, args.dl_workers)
    items = [(coco_key(s, m, u), os.path.join(args.data_dir, coco_key(s, m, u)))
             for s, m, pool in pools for u in pool
             if os.path.exists(os.path.join(args.data_dir, coco_key(s, m, u))) or coco_key(s, m, u) in store.img]
    store.add_images(items)
    tr_pool = [u for u in tr_pool if store.img.get(coco_key('train', mtr, u)) is not None]
    te_pool = [u for u in te_pool if store.img.get(coco_key('val', mte, u)) is not None]

    Dtr, Dte = coco_derive(mtr, tr_pool), coco_derive(mte, te_pool)
    store.add_texts([Dtr[u]['cap'] for u in tr_pool] + [Dte[u]['cap'] for u in te_pool])
    ifeat_tr = {u: store.img[coco_key('train', mtr, u)] for u in tr_pool}
    ifeat_te = {u: store.img[coco_key('val', mte, u)] for u in te_pool}
    P_te = CocoSplit(mte, Dte, te_pool, ifeat_te, store.txt).proposers()

    cycles = {}
    for sd in seeds:
        rng = np.random.RandomState(2000 + sd)
        perm = list(tr_pool)
        rng.shuffle(perm)
        nv = len(perm) // 6
        S_va = CocoSplit(mtr, Dtr, perm[:nv], ifeat_tr, store.txt)
        S_tr = CocoSplit(mtr, Dtr, perm[nv:], ifeat_tr, store.txt)
        P_tr, P_va = S_tr.proposers(), S_va.proposers()
        cycles[sd] = {}
        for f, _ in FAMS_COCO:
            train_c = tag_keys(sample_cycles(P_tr[f], args.n_train, rng), 'train', mtr)
            val_c = tag_keys(sample_cycles(P_va[f], args.n_val, rng), 'train', mtr)
            test_c = tag_keys(sample_cycles(P_te[f], args.n_test, rng), 'val', mte)
            cycles[sd][f] = {'train': train_c, 'val': val_c, 'test': test_c}

    texts = set()
    for sd in seeds:
        for f, _ in FAMS_COCO:
            for k in ('train', 'val', 'test'):
                for c in cycles[sd][f][k]:
                    texts.add(c[2])
                    texts.add(c[3])
                    if f == 'obj_matched':
                        texts.add(MARKER + c[2])
                        texts.add(MARKER + c[3])
    store.add_texts(texts)
    img_keys = [coco_key('train', mtr, u) for u in tr_pool] + [coco_key('val', mte, u) for u in te_pool]
    IMGr, TXTr, irow, trow = build_matrices(store, img_keys, texts)
    R, TRF, (IMG, TXT, IMG64, TXT64) = run_families(FAMS_COCO, cycles, IMGr, TXTr, irow, trow, seeds, args)

    U = defaultdict(list)
    d = IMG.shape[1]
    for sd in seeds:
        C = cycles[sd]['obj_matched']
        trd = unblocked_pairs(C['train'], irow, trow)
        vad = unblocked_pairs(C['val'], irow, trow)
        ted = unblocked_pairs(C['test'], irow, trow)
        _, tec = to_idx(C['test'], irow, trow)
        for kind in ('txt', 'add'):
            set_seed(sd)
            pr = Probe(d, d, kind).to(DEVICE)
            pr, _ = train_pairs(pr, trd, vad, IMG, TXT, IMG64, TXT64, 30, 1e-2, args.wd, args.bs)
            U[kind + '_unblocked_auroc'].append(eval_auroc(pr, ted, IMG64, TXT64))
            U[kind + '_exact_cycleacc'].append(eval_cycleacc(pr, tec, IMG64, TXT64))
    return R, TRF, U, (len(tr_pool), len(te_pool))


def nrm(s):
    return ' '.join(str(s).strip().lower().split())


def get_gqa_meta(root):
    cache = os.path.join(root, 'gqa_val_meta.pkl')
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)
    jp = os.path.join(root, 'val_sceneGraphs.json')
    if not os.path.exists(jp):
        zp = os.path.join(root, 'sceneGraphs.zip')
        if not os.path.exists(zp):
            logger.info('downloading GQA scene graphs')
            if not any(download(u, zp, retries=6, timeout=120, min_size=10 * 10 ** 6) for u in GQA_SG_URLS):
                raise RuntimeError('cannot download GQA sceneGraphs.zip (put it into %s manually)' % root)
        with zipfile.ZipFile(zp) as z:
            name = [n for n in z.namelist() if n.endswith('val_sceneGraphs.json')][0]
            src = z.open(name)
            with open(jp, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            src.close()
    with open(jp) as f:
        d = json.load(f)
    G = {}
    for iid, g in d.items():
        if not str(iid).isdigit():
            continue
        objs = g.get('objects', {})
        if len(objs) < 2:
            continue
        ol, tri = [], set()
        for _, o in objs.items():
            n = nrm(o.get('name', ''))
            attrs = tuple(sorted(set(nrm(a) for a in o.get('attributes', []))))
            ol.append((n, attrs))
            for r in o.get('relations', []):
                t = objs.get(r.get('object'))
                if t is not None:
                    tri.add((n, nrm(r.get('name', '')), nrm(t.get('name', ''))))
        G[str(iid)] = {'objs': sorted(ol), 'tri': tri}
    del d
    with open(cache, 'wb') as f:
        pickle.dump(G, f, protocol=4)
    return G


def gqa_vocab(G, ids, minc=15):
    ok = re.compile(r'^[a-z][a-z ]*[a-z]$')
    cn, ca, cp = Counter(), Counter(), Counter()
    for u in ids:
        g = G[u]
        cn.update(set(n for n, _ in g['objs']))
        ca.update(set(a for _, at in g['objs'] for a in at))
        cp.update(set(t[1] for t in g['tri']))

    def frequent(counter):
        return set(k for k, v in counter.items() if v >= minc and ok.match(k))

    return frequent(cn), frequent(ca), frequent(cp)


def acap(a, n):
    return 'a photo of %s %s %s.' % (art(a), a, n)


def tcap(t):
    return 'a photo of %s %s %s %s %s.' % (art(t[0]), t[0], t[1], art(t[2]), t[2])


def bcap(n1, c1, n2, c2):
    return 'a photo of %s %s %s and %s %s %s.' % (art(c1), c1, n1, art(c2), c2, n2)


class GqaSplit(object):
    def __init__(self, G, ids, vocab):
        nv, av, pv = vocab
        self.ids = [u for u in ids if u in G]
        self.ap_all, self.tri_all, self.ap, self.cp, self.tr, self.ucol, self.bk = {}, {}, {}, {}, {}, {}, {}
        self.i_cp, self.i_so, self.i_sp, self.i_po, self.i_spo, self.i_b = [defaultdict(list) for _ in range(6)]
        for u in self.ids:
            objs = G[u]['objs']
            apa = set((a, n) for n, at in objs for a in at)
            self.ap_all[u], self.tri_all[u] = apa, G[u]['tri']
            self.ap[u] = sorted(x for x in apa if x[0] in av and x[1] in nv)
            self.cp[u] = [x for x in self.ap[u] if x[0] in COLORS]
            self.tr[u] = sorted(t for t in G[u]['tri'] if t[0] in nv and t[2] in nv and t[1] in pv and t[0] != t[2])
            cnt = Counter(n for n, _ in objs)
            uc = {}
            for n, at in objs:
                if cnt[n] == 1 and n in nv:
                    cols = set(a for a in at if a in COLORS)
                    if cols:
                        uc[n] = cols
            self.ucol[u] = uc
            bk = []
            nm = sorted(uc)
            for i in range(len(nm)):
                for j in range(i + 1, len(nm)):
                    for c1 in sorted(uc[nm[i]]):
                        for c2 in sorted(uc[nm[j]]):
                            if c1 != c2:
                                bk.append((nm[i], c1, nm[j], c2))
            self.bk[u] = bk
            for (c, n) in self.cp[u]:
                self.i_cp[n].append((u, c))
            for t in self.tr[u]:
                s, p, o = t
                self.i_so[(s, o)].append((u, p))
                self.i_sp[(s, p)].append((u, o))
                self.i_po[(p, o)].append((u, s))
                self.i_spo[t].append(u)
            for k in bk:
                self.i_b[k].append(u)
        self.w_ap = [u for u in self.ids if self.ap[u]]
        self.w_cp = [u for u in self.ids if self.cp[u]]
        self.w_tr = [u for u in self.ids if self.tr[u]]
        self.w_bk = [u for u in self.ids if self.bk[u]]

    def p_AD(self, g):
        u1 = pick(g, self.w_cp)
        if u1 is None:
            return None
        c1, n = pick(g, self.cp[u1])
        u2, c2 = pick(g, self.i_cp[n])
        if u2 == u1 or c2 == c1 or (c2, n) in self.ap_all[u1] or (c1, n) in self.ap_all[u2]:
            return None
        return (u1, u2, acap(c1, n), acap(c2, n))

    def p_CD(self, g):
        u1, u2 = pick(g, self.w_ap), pick(g, self.w_ap)
        if u1 is None or u1 == u2:
            return None
        a1, n1 = pick(g, self.ap[u1])
        a2, n2 = pick(g, self.ap[u2])
        if n1 == n2 or (a2, n2) in self.ap_all[u1] or (a1, n1) in self.ap_all[u2]:
            return None
        return (u1, u2, acap(a1, n1), acap(a2, n2))

    def p_SR(self, g):
        u1, u2 = pick(g, self.w_tr), pick(g, self.w_tr)
        if u1 is None or u1 == u2:
            return None
        t1, t2 = pick(g, self.tr[u1]), pick(g, self.tr[u2])
        if t1[0] == t2[0] or t1[2] == t2[2] or t2 in self.tri_all[u1] or t1 in self.tri_all[u2]:
            return None
        return (u1, u2, tcap(t1), tcap(t2))

    def _rel(self, g, mode):
        u1 = pick(g, self.w_tr)
        if u1 is None:
            return None
        s, p, o = pick(g, self.tr[u1])
        if mode == 'RS':
            x = pick(g, self.i_so[(s, o)])
            t2 = (s, x[1], o) if x else None
        elif mode == 'OP':
            x = pick(g, self.i_sp[(s, p)])
            t2 = (s, p, x[1]) if x else None
        else:
            x = pick(g, self.i_po[(p, o)])
            t2 = (x[1], p, o) if x else None
        if x is None:
            return None
        u2, t1 = x[0], (s, p, o)
        if u2 == u1 or t2 == t1 or t2 in self.tri_all[u1] or t1 in self.tri_all[u2]:
            return None
        return (u1, u2, tcap(t1), tcap(t2))

    def p_SS(self, g):
        u1 = pick(g, self.w_tr)
        if u1 is None:
            return None
        s, p, o = pick(g, self.tr[u1])
        u2 = pick(g, self.i_spo.get((o, p, s), []))
        if u2 is None or u2 == u1 or (o, p, s) in self.tri_all[u1] or (s, p, o) in self.tri_all[u2]:
            return None
        return (u1, u2, tcap((s, p, o)), tcap((o, p, s)))

    def p_BS(self, g):
        u1 = pick(g, self.w_bk)
        if u1 is None:
            return None
        n1, c1, n2, c2 = pick(g, self.bk[u1])
        u2 = pick(g, self.i_b.get((n1, c2, n2, c1), []))
        if u2 is None or u2 == u1:
            return None
        a, b = self.ucol[u1], self.ucol[u2]
        if c2 in a[n1] or c1 in a[n2] or c1 in b[n1] or c2 in b[n2]:
            return None
        return (u1, u2, bcap(n1, c1, n2, c2), bcap(n1, c2, n2, c1))

    def proposers(self):
        return {'AD': self.p_AD, 'CD': self.p_CD, 'SR': self.p_SR,
                'RS': lambda g: self._rel(g, 'RS'), 'OP': lambda g: self._rel(g, 'OP'),
                'OO': lambda g: self._rel(g, 'OO'), 'SS': self.p_SS, 'BS': self.p_BS}


def gqa_key(u):
    return 'gqa/images/%s.jpg' % u


def gqa_experiments(store, seeds, args):
    root = os.path.join(args.data_dir, 'gqa')
    if not os.path.exists(root):
        os.makedirs(root)
    G = get_gqa_meta(root)
    elig = sorted(u for u in G if len(G[u]['objs']) >= 3)
    rs = np.random.RandomState(777)
    rs.shuffle(elig)
    pool = elig[:args.gqa_pool]
    jobs = [(gqa_key(u), [x % u for x in VG_URLS], os.path.join(args.data_dir, gqa_key(u)))
            for u in pool if store.img.get(gqa_key(u)) is None]
    logger.info('downloading %d Visual Genome images', len(jobs))
    download_many(jobs, args.dl_workers)
    store.add_images([(gqa_key(u), os.path.join(args.data_dir, gqa_key(u))) for u in pool
                      if os.path.exists(os.path.join(args.data_dir, gqa_key(u))) or gqa_key(u) in store.img])
    pool = [u for u in pool if store.img.get(gqa_key(u)) is not None]
    if len(pool) < 1500:
        raise RuntimeError('too few GQA images downloaded (%d)' % len(pool))
    vocab = gqa_vocab(G, pool)
    cycles = {}
    for sd in seeds:
        rng = np.random.RandomState(3000 + sd)
        perm = list(pool)
        rng.shuffle(perm)
        nte, nva = int(0.3 * len(perm)), int(0.1 * len(perm))
        P = {k: GqaSplit(G, ids, vocab).proposers() for k, ids in
             (('test', perm[:nte]), ('val', perm[nte:nte + nva]), ('train', perm[nte + nva:]))}
        cycles[sd] = {}
        for f, _ in FAMS_GQA:
            cycles[sd][f] = {}
            for k, n in (('train', args.n_train), ('val', args.n_val), ('test', args.n_test)):
                cycles[sd][f][k] = [(gqa_key(a), gqa_key(b), v1, v2)
                                    for (a, b, v1, v2) in sample_cycles(P[k][f], n, rng)]
    texts = set(c[j] for sd in seeds for f, _ in FAMS_GQA for k in ('train', 'val', 'test')
                for c in cycles[sd][f][k] for j in (2, 3))
    store.add_texts(texts)
    IMGr, TXTr, irow, trow = build_matrices(store, [gqa_key(u) for u in pool], texts)
    R, TRF, _ = run_families(FAMS_GQA, cycles, IMGr, TXTr, irow, trow, seeds, args)
    return R, TRF, len(pool)


def fam_table(L, R, fams):
    L.append('family            grp      n(tr/va/te)      TV     auditDev  CLIP   diag   full   r1     r4     r16    r64')
    for f, grp in fams:
        r = R[f]
        if not r.get('n'):
            L.append('%-17s %-8s skipped (too few cycles: %s)' % (f, grp, r.get('skipped')))
            continue
        n = np.mean(np.array(r['n']), 0).astype(int)
        L.append('%-17s %-8s %5d/%4d/%4d  %.3f  %.1e  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f' % (
            f, grp, n[0], n[1], n[2], max(r['tv']), max(r['audit_dev']), nanmean(r['clip']), nanmean(r['diag']),
            nanmean(r['full']), nanmean(r['r1']), nanmean(r['r4']), nanmean(r['r16']), nanmean(r['r64'])))


def group_summary(L, R, fams):
    groups = []
    for _, g in fams:
        if g not in groups:
            groups.append(g)
    for g in groups:
        fs = [f for f, gg in fams if gg == g and R[f].get('n')]
        if not fs:
            L.append('%-8s: no families available' % g)
            continue
        cl = [nanmean(R[f]['clip']) for f in fs]
        ip = [nanmean(R[f][k]) for f in fs for k in ('diag', 'full', 'r64')]
        ad = max(max(R[f]['audit_dev']) for f in fs)
        L.append('%-8s (%d fams): additive audit maxDev=%.1e | CLIP %.3f-%.3f | trained probes %.3f-%.3f'
                 % (g, len(fs), ad, min(cl), max(cl), min(ip), max(ip)))


def report_synthetic(L, S):
    sv = S['spectrum_sv']
    L.append('')
    L.append('--- A. Synthetic controls (AUROC, unblocked two-sample) ---')
    L.append('Pure interaction   marginal(additive) : %s' % fmt_mean_std(S['pure_marginal']))
    L.append('                   bilinear           : %s' % fmt_mean_std(S['pure_bilinear']))
    L.append('Marginal shortcut  marginal(additive) : %s' % fmt_mean_std(S['shortcut_marginal']))
    L.append('Rank bottleneck    marginal(additive) : %s' % fmt_mean_std(S['rank_marginal']))
    for r in (1, 2, 4, 8, 16):
        frac = np.sqrt((sv[:r] ** 2).sum() / (sv ** 2).sum())
        L.append('                   rank %-2d interaction: %s   spectrum-predicted capture=%.3f' % (
            r, fmt_mean_std(S['rank_r%d' % r]), frac))
    L.append('Binding            bag features       : %s' % fmt_mean_std(S['bind_bag']))
    L.append('                   pair-aware features: %s' % fmt_mean_std(S['bind_pair']))


def report_coco(L, R, TRF, U, npool):
    L.append('')
    L.append('--- B. COCO raw-image exact cycles (CycleAcc mean over seeds; train2017 pool %d imgs, '
             'test=val2017 pool %d imgs) ---' % npool)
    fam_table(L, R, FAMS_COCO)
    L.append('group summary:')
    group_summary(L, R, FAMS_COCO)
    L.append('')
    L.append('--- C. Strict COCO relations: CLIP cosine | rank-64 bilinear ---')
    for f in STRICT_FAMS:
        if R[f].get('n'):
            L.append('%-18s %s | %s' % (f, fmt_mean_std(R[f]['clip'], 3), fmt_mean_std(R[f]['r64'], 3)))
    L.append('')
    L.append('--- D. Rank sweep: rank 1 / 4 / 16 / 64 ---')
    for f, name in RANK_SWEEP_FAMS:
        if R[f].get('n'):
            L.append('%-22s %s | %s | %s | %s' % (name, fmt_mean_std(R[f]['r1'], 3), fmt_mean_std(R[f]['r4'], 3),
                                                  fmt_mean_std(R[f]['r16'], 3), fmt_mean_std(R[f]['r64'], 3)))
    L.append('')
    L.append('--- E. Unblocked text-marginal contamination on obj_matched ---')
    L.append('text-only probe : unblocked AUROC %s | exact CycleAcc %s'
             % (fmt_mean_std(U['txt_unblocked_auroc']), fmt_mean_std(U['txt_exact_cycleacc'])))
    L.append('additive probe  : unblocked AUROC %s | exact CycleAcc %s'
             % (fmt_mean_std(U['add_unblocked_auroc']), fmt_mean_std(U['add_exact_cycleacc'])))
    L.append('')
    L.append('--- F. COCO transfer alignment (rank-64 trained on family i, tested on j, off-diagonal) ---')
    L.append('pairs/seed=%d  Pearson=%s  Spearman=%s'
             % (TRF['npairs'][0] if TRF['npairs'] else 0, fmt_mean_std(TRF['pearson'], 3),
                fmt_mean_std(TRF['spearman'], 3)))


def report_gqa(L, RG, TRG, ng):
    L.append('--- G. GQA scene-graph exact cycles (val_sceneGraphs split image-disjoint 60/10/30, %d imgs) ---' % ng)
    fam_table(L, RG, FAMS_GQA)
    L.append('group summary:')
    group_summary(L, RG, FAMS_GQA)
    L.append('GQA transfer alignment: Pearson=%s  Spearman=%s'
             % (fmt_mean_std(TRG['pearson'], 3), fmt_mean_std(TRG['spearman'], 3)))


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    L = ['=== Counterfactual Pairing Cycles: exact-cycle diagnostics ===',
         'config: CLIP ViT-B/32 frozen; probes trained with cycle loss, %d epochs, bs=%d, lr=%g, wd=%g; seeds=%s; '
         'cycles/family train/val/test<=%d/%d/%d; device=%s' % (args.epochs, args.bs, args.lr, args.wd, args.seeds,
                                                                 args.n_train, args.n_val, args.n_test, DEVICE)]
    store = FeatStore(os.path.join(args.data_dir, 'clip_feat_store.pkl'),
                      os.path.join(args.data_dir, 'clip-vit-base-patch32'), args.num_workers)
    try:
        if not args.skip_synth:
            synth_seeds = [int(s) for s in args.synth_seeds.split(',')]
            report_synthetic(L, synthetic_controls(synth_seeds, args.wd, args.bs))
        R, TRF, U, npool = coco_experiments(store, seeds, args)
        report_coco(L, R, TRF, U, npool)
    except Exception as e:
        L.append('COCO/synthetic experiments failed: %s: %s' % (type(e).__name__, e))

    L.append('')
    if args.skip_gqa:
        L.append('--- G. GQA: skipped ---')
    else:
        try:
            RG, TRG, ng = gqa_experiments(store, seeds, args)
            report_gqa(L, RG, TRG, ng)
        except Exception as e:
            L.append('--- G. GQA: failed: %s: %s' % (type(e).__name__, e))

    report = '\n'.join(L)
    with open(args.log, 'w') as f:
        f.write(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
