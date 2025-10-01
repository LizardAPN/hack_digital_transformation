import os
import math
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from data.datasets import GeoDataset
from models.loss_function import contrastive_loss_with_queue
from models.metrics import geo_metrics_from_indices
import faiss

def train_and_evaluate(
    image_encoder: torch.nn.Module,
    loc_encoder: torch.nn.Module,
    train_csv: str,
    test_csv: str,
    out_dir: str = './checkpoints',
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 3e-5,
    device: Optional[torch.device] = None,
    eval_k: int = 10,
    radii_m: List[float] = [25.0, 50.0, 100.0],
    use_queue: bool = False,
    queue_size: int = 65536,
    amp: bool = True,
    log_interval: int = 10
) -> Dict[str, object]:

    os.makedirs(out_dir, exist_ok=True)
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    preprocess = getattr(image_encoder, 'preprocess_image', None)

    train_ds = GeoDataset(train_csv, preprocess=preprocess)
    test_ds = GeoDataset(test_csv, preprocess=preprocess)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=(device.type == 'cuda'))

    if isinstance(image_encoder, torch.nn.Module):
        image_encoder = image_encoder.to(device)
        img_forward = lambda x: image_encoder(x)
    else:
        img_forward = image_encoder
    loc_encoder = loc_encoder.to(device)

    sample_img, sample_gps, _ = train_ds[0]
    sample_inp = sample_img.unsqueeze(0).to(device)

    with torch.no_grad():
        img_emb_dim = img_forward(sample_inp).shape[1]
        loc_emb_dim = loc_encoder(sample_gps.unsqueeze(0).to(device)).shape[1]
    if img_emb_dim != loc_emb_dim:
        raise RuntimeError(f"Image and location encoders must produce embeddings of the same dimension: {img_emb_dim} vs {loc_emb_dim}")

    emb_dim = img_emb_dim
    queue = None
    if use_queue:
        queue = np.zeros((0, emb_dim), dtype=np.float32)

    params = []
    if isinstance(image_encoder, torch.nn.Module):
        params += list(image_encoder.parameters())
    params += list(loc_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-6)

    # AMP setup
    use_amp = amp and device.type == 'cuda'
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = torch.amp.GradScaler(enabled=use_amp and device.type == device_type)

    loss_history: List[float] = []
    for epoch in range(1, epochs + 1):
        if isinstance(image_encoder, torch.nn.Module):
            image_encoder.train()
        loc_encoder.train()
        running_loss = 0.0
        examples_seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch_idx, (imgs, gps, paths) in enumerate(pbar, 1):
            imgs = imgs.to(device)
            gps = gps.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                im_vecs = img_forward(imgs)
                loc_vecs = loc_encoder(gps)
                if use_queue and queue is not None and len(queue) > 0:
                    q_tensor = torch.from_numpy(queue).to(device)
                else:
                    q_tensor = None
                loss = contrastive_loss_with_queue(im_vecs, loc_vecs, q_tensor, temperature=0.07)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_actual = imgs.size(0)
            running_loss += float(loss.item()) * batch_size_actual
            examples_seen += batch_size_actual

            if use_queue:
                with torch.no_grad():
                    loc_norm = F.normalize(loc_vecs, p=2, dim=-1).cpu().numpy()
                if queue is None or queue.size == 0:
                    queue = loc_norm.copy()
                else:
                    queue = np.concatenate([queue, loc_norm], axis=0)
                    if queue.shape[0] > queue_size:
                        queue = queue[-queue_size:]

            if batch_idx % log_interval == 0:
                mean_loss = running_loss / max(1, examples_seen)
                pbar.set_postfix(mean_loss=f"{mean_loss:.6f}")
                print(f"Epoch {epoch}, batch {batch_idx}, mean loss so far: {mean_loss:.6f}")
        pbar.close()

        epoch_loss = running_loss / max(1, examples_seen)
        print(f"Epoch {epoch} completed, average loss: {epoch_loss:.6f}")
        loss_history.append(epoch_loss)
        ckpt = {
            'epoch': epoch,
            'image_encoder': image_encoder.state_dict() if isinstance(image_encoder, torch.nn.Module) else None,
            'loc_encoder': loc_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'queue': queue
        }
        torch.save(ckpt, os.path.join(out_dir, f'checkpoint_epoch{epoch}.pth'))

    df_train = pd.read_csv(train_csv)[['lat', 'lon']].reset_index(drop=True)
    df_test = pd.read_csv(test_csv)[['lat', 'lon']].reset_index(drop=True)

    with torch.no_grad():
        if isinstance(image_encoder, torch.nn.Module):
            image_encoder.eval()
        loc_encoder.eval()

        train_coords = df_train.to_numpy(dtype=np.float32)
        loc_embs_list = []
        for i in tqdm(range(0, train_coords.shape[0], 1024), desc="Encoding train coords", unit="block"):
            block = torch.from_numpy(train_coords[i:i + 1024]).to(device)
            feats = loc_encoder(block)
            loc_embs_list.append(F.normalize(feats, p=2, dim=-1).cpu().numpy().astype(np.float32))
        loc_embs = np.vstack(loc_embs_list) if len(loc_embs_list) > 0 else np.zeros((0, emb_dim), dtype=np.float32)

        test_image_paths = test_ds.paths
        image_embs_list = []
        for i in tqdm(range(0, len(test_image_paths), 128), desc="Encoding test images", unit="batch"):
            batch_paths = test_image_paths[i:i + 128]
            tensors = []
            for p in batch_paths:
                img = Image.open(p).convert('RGB')
                if preprocess is not None:
                    t = preprocess(img)
                    if isinstance(t, torch.Tensor) and t.ndim == 4:
                        t = t.squeeze(0)
                else:
                    from torchvision import transforms as T
                    t = T.Compose([
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
                    ])(img)
                tensors.append(t)
            batch_t = torch.stack(tensors, dim=0).to(device)
            feats = img_forward(batch_t)
            image_embs_list.append(F.normalize(feats, p=2, dim=-1).cpu().numpy().astype(np.float32))
        image_embs = np.vstack(image_embs_list) if len(image_embs_list) > 0 else np.zeros((0, emb_dim), dtype=np.float32)

    dim = loc_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(loc_embs)
    K = min(eval_k, loc_embs.shape[0])
    _, I = index.search(image_embs, K)

    metrics = geo_metrics_from_indices(df_test, df_train, I, ks=[1, 5, 10], radii_m=radii_m)
    return {
        'loss_history': loss_history,
        'last_metrics': metrics
    }