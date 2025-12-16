import argparse
import os
import sys
import time
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy import sparse
import gc


class SophiaG(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=1e-1, *, maximize: bool = False,
                 capturable: bool = False, dynamic: bool = False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho parameter at index 1: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay,
                        maximize=maximize, capturable=capturable, dynamic=dynamic)
        super(SophiaG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
            group.setdefault('dynamic', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros(
                        (1,), dtype=torch.float, device=p.device) if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                state['hessian'].mul_(beta2).addcmul_(
                    p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def update_exp_avg(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['exp_avg'].mul_(beta1).add_(p.grad, alpha=1 - beta1)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.update_hessian()
        self.update_exp_avg()
        for group in self.param_groups:
            params_with_grad, grads, exp_avgs, state_steps, hessian = [], [], [], [], []
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros(
                        (1,), dtype=torch.float, device=p.device) if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                exp_avgs.append(state['exp_avg'])
                state_steps.append(state['step'])
                hessian.append(state['hessian'])
                if self.defaults['capturable']:
                    bs = torch.ones((1,), dtype=torch.float,
                                    device=p.device) * bs
            self._sophiag(params_with_grad, grads, exp_avgs, hessian, state_steps, group['capturable'],
                          bs=bs, beta1=beta1, beta2=beta2, rho=group['rho'], lr=group['lr'],
                          weight_decay=group['weight_decay'], maximize=group['maximize'])
        return loss

    def _sophiag(self, params, grads, exp_avgs, hessian, state_steps, capturable, *, bs, beta1, beta2, rho, lr, weight_decay, maximize):
        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            exp_avg = exp_avgs[i]
            hess = hessian[i]
            step_t = state_steps[i]
            step_t += 1
            param.mul_(1 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            step_size_neg = -lr
            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)


class MLPModel(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.activation = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(input_features)
        self.fc1 = nn.Linear(input_features, 800)
        self.ln1 = nn.LayerNorm(800, elementwise_affine=True)
        self.bn2 = nn.BatchNorm1d(800)
        self.fc2 = nn.Linear(800, 600)
        self.ln2 = nn.LayerNorm(600, elementwise_affine=True)
        self.bn3 = nn.BatchNorm1d(600)
        self.fc3 = nn.Linear(600, 400)
        self.ln3 = nn.LayerNorm(400, elementwise_affine=True)
        self.bn4 = nn.BatchNorm1d(1200)
        self.fc4 = nn.Linear(1200, output_features)
        self.ln4 = nn.LayerNorm(output_features, elementwise_affine=True)
        self.sigm = nn.Sigmoid()

    def forward(self, inputs):
        fc1_out = self.activation(self.ln1(self.fc1(self.bn1(inputs))))
        x = self.activation(self.ln2(self.fc2(self.bn2(fc1_out))))
        x = self.activation(self.ln3(self.fc3(self.bn3(x))))
        x = torch.cat([x, fc1_out], axis=-1)
        x = self.ln4(self.fc4(self.bn4(x)))
        return self.sigm(x)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str, required=True)
parser.add_argument('-m', '--model-name', type=str, default="nn_pMLP")
parser.add_argument('-d', '--device', type=str, default="0")
parser.add_argument('--fold', type=int, required=True)

if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device(f"cuda:0")

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    T5_DIR = os.path.join(config['base_path'], config['embeds_path'], 't5')
    ESM2_DIR = os.path.join(config['base_path'],
                            config['embeds_path'], 'esm_small')
    FEAT_DIR = os.path.join(config['base_path'],
                            config['helpers_path'], 'feats')
    FOLDS_DIR = os.path.join(config['base_path'], config['helpers_path'])

    MODEL_OUT_DIR = os.path.join(
        config['base_path'], config['models_path'], args.model_name)
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)

    print(f"--- NN Worker: Fold {args.fold} on GPU {args.device} ---")

    def load_embeds(path):
        return np.load(path).astype(np.float32)

    print("Loading T5...")
    t5_train = load_embeds(os.path.join(T5_DIR, 'train_embeds.npy'))
    t5_test = load_embeds(os.path.join(T5_DIR, 'test_embeds.npy'))

    print("Loading ESM2...")
    esm_train = load_embeds(os.path.join(ESM2_DIR, 'train_embeds.npy'))
    esm_test = load_embeds(os.path.join(ESM2_DIR, 'test_embeds.npy'))

    X = np.concatenate([t5_train, esm_train], axis=1)
    X_test = np.concatenate([t5_test, esm_test], axis=1)

    del t5_train, esm_train, t5_test, esm_test
    gc.collect()

    print("Loading Targets...")
    Y = sparse.load_npz(os.path.join(FEAT_DIR, 'Y_31466_sparse_float32.npz'))
    Y = Y.toarray()[:, :2000]

    folds = np.load(os.path.join(FOLDS_DIR, 'folds_gkf.npy'))

    f = args.fold
    ix_train = np.where(folds != f)[0]
    ix_val = np.where(folds == f)[0]

    X_train, Y_train = X[ix_train], Y[ix_train]
    X_val = X[ix_val]

    print(f"Fold {f}: Train shape {X_train.shape}, Val shape {X_val.shape}")

    model = MLPModel(X.shape[1], Y.shape[1]).to(device)

    BATCH_SIZE = 128
    EPOCHS = 20
    optimizer = SophiaG(model.parameters(), lr=0.001, betas=(
        0.965, 0.99), rho=0.01, weight_decay=1e-1)
    criterion = nn.BCELoss()

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print("Training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(train_loader):.5f}")

    torch.save(model.state_dict(), os.path.join(
        MODEL_OUT_DIR, f'model_fold_{f}.pt'))

    print("Inference...")
    model.eval()

    def predict_batch(data_np, batch_size=5120):
        preds = []
        with torch.no_grad():
            for i in range(0, len(data_np), batch_size):
                batch = torch.tensor(
                    data_np[i:i+batch_size], dtype=torch.float32).to(device)
                preds.append(model(batch).cpu().numpy())
        return np.concatenate(preds)

    val_preds = predict_batch(X_val)
    oof_full = np.zeros((X.shape[0], Y.shape[1]), dtype=np.float16)
    oof_full[ix_val] = val_preds.astype(np.float16)

    np.save(os.path.join(MODEL_OUT_DIR, f'temp_oof_fold_{f}.npy'), oof_full)

    test_preds = predict_batch(X_test).astype(np.float16)
    np.save(os.path.join(MODEL_OUT_DIR, f'temp_test_fold_{f}.npy'), test_preds)

    print(f"✅ Finished Fold {f}")
