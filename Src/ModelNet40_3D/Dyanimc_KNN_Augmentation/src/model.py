
# ─────────────────────────────────────────────────────
# 8.  TRAINING LOOP
# ─────────────────────────────────────────────────────
def apply_augmentation(batch_data):
    """
    Apply augmentation to each sample in a batch IN-PLACE.
    Called inside train_epoch so augmentation is different every epoch.
    Only modifies .pos and updates .x accordingly (re-concatenate norm).
    """
    for data in batch_data.to_data_list():
        aug_pos = augment_pointcloud(data.pos)
        data.pos = aug_pos
        # Rebuild x = [aug_pos | norm | triangle_count]
        # norm (cols 3-5) and triangle count (col 6) are UNCHANGED by augmentation
        data.x = torch.cat([aug_pos, data.x[:, 3:]], dim=1)
    return Batch.from_data_list(batch_data.to_data_list())
 
 
# We need Batch for apply_augmentation
from torch_geometric.data import Batch
 
# num_workers=0 — Kaggle notebooks fork new processes inside an already
# forked process which breaks Python's multiprocessing assertions.
# num_workers=0 runs data loading in the main process, slightly slower
# but stable. On P100 the GPU is the bottleneck anyway, not data loading.
train_loader = DataLoader(train_list, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_list,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)
 
model     = MotifGINBaseline(in_channels=7, hidden=128,
                              num_classes=NUM_CLASSES, k=K_NEIGHBORS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
 
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters : {total_params:,}")
print(f"Device           : {DEVICE}")
print(f"Epochs           : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
 
 
def train_epoch(model, loader):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
 
    for batch in tqdm(loader, desc='  train', leave=False):
        # Apply augmentation on the CPU before moving to device
        batch = apply_augmentation(batch)
        batch = batch.to(DEVICE)
 
        optimizer.zero_grad()
        out  = model(batch)
        loss = F.cross_entropy(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item() * batch.num_graphs
        correct    += out.argmax(1).eq(batch.y.squeeze()).sum().item()
        total      += batch.num_graphs
 
    return total_loss / total, correct / total
 
 
@torch.no_grad()
def test_epoch(model, loader):
    model.eval()
    all_preds, all_labels = [], []
 
    for batch in tqdm(loader, desc='  test ', leave=False):
        batch  = batch.to(DEVICE)
        preds  = model(batch).argmax(1).cpu().numpy()
        labels = batch.y.squeeze().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
 
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
 
    return {
        'OA'       : accuracy_score(all_labels, all_preds) * 100,
        'mAcc'     : balanced_accuracy_score(all_labels, all_preds) * 100,
        'macro_f1' : f1_score(all_labels, all_preds,
                               average='macro', zero_division=0) * 100,
    }
 
 
# ── Resume from checkpoint if it exists ──────────────────────────────
CKPT_PATH   = '/kaggle/working/best_motif_gin.pt'
START_EPOCH = 1
best_oa     = 0.0
history     = {'train_loss': [], 'train_acc': [], 'OA': [], 'mAcc': [], 'macro_f1': []}
 
if os.path.exists(CKPT_PATH):
    print(f"Checkpoint found — resuming from {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
 
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        # Full checkpoint — weights + optimizer + scheduler + epoch + history
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        START_EPOCH = ckpt['epoch'] + 1
        best_oa     = ckpt['best_oa']
        history     = ckpt['history']
        print(f"  Resumed from epoch {ckpt['epoch']}  |  best OA so far: {best_oa:.2f}%")
        print(f"  Continuing from epoch {START_EPOCH} → {EPOCHS}")
    else:
        # Plain state_dict only (saved by original code before this fix)
        model.load_state_dict(ckpt)
        START_EPOCH = 31   # crashed around epoch 30
        # Fast-forward cosine scheduler so LR matches where we left off
        for _ in range(START_EPOCH - 1):
            scheduler.step()
        print(f"  Loaded weights only (plain state_dict)")
        print(f"  Assuming crashed at epoch 30 — resuming from epoch {START_EPOCH}")
        print(f"  LR fast-forwarded to: {optimizer.param_groups[0]['lr']:.6f}")
else:
    print("No checkpoint found — training from scratch")
 
print(f"\nTraining from epoch {START_EPOCH} to {EPOCHS} ...")
for epoch in trange(START_EPOCH, EPOCHS + 1, desc='Epochs'):
    tr_loss, tr_acc = train_epoch(model, train_loader)
    metrics         = test_epoch(model, test_loader)
    scheduler.step()
 
    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc * 100)
    for k in ['OA', 'mAcc', 'macro_f1']:
        history[k].append(metrics[k])
 
    if metrics['OA'] > best_oa:
        best_oa = metrics['OA']
        torch.save({
            'state_dict' : model.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            'scheduler'  : scheduler.state_dict(),
            'epoch'      : epoch,
            'best_oa'    : best_oa,
            'history'    : history,
        }, '/kaggle/working/best_motif_gin.pt')
 
    if epoch % 5 == 0:
        print(f"  Epoch {epoch:3d} | loss {tr_loss:.4f} | "
              f"trAcc {tr_acc*100:.1f}% | "
              f"OA {metrics['OA']:.1f}% | "
              f"mAcc {metrics['mAcc']:.1f}% | "
              f"F1 {metrics['macro_f1']:.1f}%")
 
print(f"\nBest OA : {best_oa:.2f}%")
print(f"Baseline comparisons:")
print(f"  Your GCN (50 ep)  → OA ~{0.0:.1f}%  (fill in from your run)")
print(f"  PointNet           → OA 89.2%  mAcc 86.2%")
print(f"  DGCNN              → OA 92.9%  mAcc 90.2%")
print(f"  This model target  → OA 88-91% with motif features + GIN")
 
 
# ─────────────────────────────────────────────────────
# 9.  QUICK TRAINING CURVE PLOT
# ─────────────────────────────────────────────────────
import matplotlib.pyplot as plt
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
axes[0].plot(history['train_loss'], label='Train loss')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-entropy loss')
axes[0].set_title('Training Loss'); axes[0].legend()
 
axes[1].plot(history['train_acc'], label='Train acc')
axes[1].plot(history['OA'],        label='Test OA')
axes[1].plot(history['mAcc'],      label='Test mAcc')
axes[1].plot(history['macro_f1'],  label='Test F1')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy Curves'); axes[1].legend()
 
plt.tight_layout()
plt.savefig('/kaggle/working/motif_gin_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: motif_gin_curves.png")