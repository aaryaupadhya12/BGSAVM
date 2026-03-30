class GCNModel(nn.Module):
    def __init__(self, in_dim=EMBED_DIM, hidden_dim=256, num_classes=10, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.head(x)


class GINModel(nn.Module):
    def __init__(self, in_dim=EMBED_DIM, hidden_dim=256, num_classes=10, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        mlp_in = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.convs.append(GINConv(mlp_in))
        self.norms.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.head(x)


def train_gnn(model_name='GCN', alpha=0.5, num_layers=3,
              epochs=50, lr=1e-3, batch_size=64):

    print(f"\n{'='*55}")
    print(f"{model_name} | pretrained ViT-Tiny | alpha={alpha} | layers={num_layers} | device={DEVICE}")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE} = {NUM_PATCHES} patches | embed_dim={EMBED_DIM}")
    print(f"{'='*55}")

    train_loader, test_loader = get_loaders(batch_size)

    vit = load_pretrained_vit()
    vit.eval()
    for p in vit.parameters():
        p.requires_grad = False

    graph_builder = SpatialGraph(grid_size=GRID_SIZE, alpha=alpha).to(DEVICE)

    if model_name == 'GCN':
        model = GCNModel(in_dim=EMBED_DIM, hidden_dim=256, num_classes=10, num_layers=num_layers).to(DEVICE)
    else:
        model = GINModel(in_dim=EMBED_DIM, hidden_dim=256, num_classes=10, num_layers=num_layers).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    opt       = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history  = {'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[]}
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, correct, total = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Ep {ep:03d}/{epochs} [Train]", leave=False, ncols=100)
        for x, y in train_bar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                features = extract_patch_features(vit, x)
                adj      = graph_builder(features)
            pyg_batch = build_pyg_graph(features, adj).to(DEVICE)

            opt.zero_grad()
            out  = model(pyg_batch)
            loss = criterion(out, y)
            loss.backward()
            opt.step()

            tr_loss += loss.item()
            correct += out.argmax(1).eq(y).sum().item()
            total   += y.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

        scheduler.step()
        tr_loss /= len(train_loader)
        tr_acc   = 100.0 * correct / total

        model.eval()
        te_loss, correct, total = 0.0, 0, 0
        test_bar = tqdm(test_loader, desc=f"Ep {ep:03d}/{epochs} [Test] ", leave=False, ncols=100)
        with torch.no_grad():
            for x, y in test_bar:
                x, y = x.to(DEVICE), y.to(DEVICE)
                features  = extract_patch_features(vit, x)
                adj       = graph_builder(features)
                pyg_batch = build_pyg_graph(features, adj).to(DEVICE)
                out  = model(pyg_batch)
                loss = criterion(out, y)
                te_loss += loss.item()
                correct += out.argmax(1).eq(y).sum().item()
                total   += y.size(0)
                test_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

        te_loss /= len(test_loader)
        te_acc   = 100.0 * correct / total

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), f'{model_name}_pretrained_best.pth')

        print(f"Ep {ep:03d}/{epochs} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.2f}% | te_loss={te_loss:.4f} te_acc={te_acc:.2f}%")

    print(f"Best test acc: {best_acc:.2f}%")
    return history, best_acc