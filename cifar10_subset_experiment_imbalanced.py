"""
CIFAR-10 Imbalanced Random Subset Selection Experiment
=======================================================
Creates a long-tailed imbalanced CIFAR-10 training set, then randomly
samples a subset of 5k images from it 50 times. Each trial trains a
fresh SmallResNet and evaluates on the full (balanced) test set.

Reports mean, min, max, std, median of test accuracy + histogram.

Usage
-----
    python cifar10_imbalanced_random_experiment.py

    python cifar10_imbalanced_random_experiment.py \
        --imb_type exp \
        --imb_factor 0.01 \
        --subset_size 5000 \
        --num_trials 50 \
        --epochs_per_trial 10
"""

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import wandb


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = dict(
    subset_size=5000,
    num_trials=50,
    epochs_per_trial=100,
    batch_size=128,
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    seed=42,
    imb_type="exp",
    imb_factor=0.01,
    wandb_project="cifar10-imbalanced-random",
)


# ──────────────────────────────────────────────────────────────────────────────
# Imbalanced CIFAR-10
# ──────────────────────────────────────────────────────────────────────────────
class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type="exp", imb_factor=0.01, rand_number=0,
                 train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == "exp":
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == "step":
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data, new_targets = [], []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes    = np.unique(targets_np)
        self.num_per_cls_dict = {}
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class] * the_img_num)
        self.data    = np.vstack(new_data)
        self.targets = new_targets

    def get_cls_num_list(self):
        return [self.num_per_cls_dict[i] for i in range(self.cls_num)]


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ConvBlock(32, 64,  stride=2)
        self.layer2 = ConvBlock(64, 128, stride=2)
        self.layer3 = ConvBlock(128, 256, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.fc(self.pool(x).flatten(1))


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])


def get_datasets(config):
    train_full = IMBALANCECIFAR10(
        root="./data",
        imb_type=config["imb_type"],
        imb_factor=config["imb_factor"],
        train=True,
        transform=TRAIN_TRANSFORM,
        download=True,
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=TEST_TRANSFORM
    )
    return train_full, test_set


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out  = model(inputs)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct    += out.argmax(1).eq(targets).sum().item()
        total      += inputs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        out  = model(inputs)
        loss = criterion(out, targets)
        total_loss += loss.item() * inputs.size(0)
        correct    += out.argmax(1).eq(targets).sum().item()
        total      += inputs.size(0)
    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Single trial
# ──────────────────────────────────────────────────────────────────────────────
def run_trial(trial_idx, train_full, test_loader, config, device):
    indices      = random.sample(range(len(train_full)), config["subset_size"])
    subset       = Subset(train_full, indices)
    train_loader = DataLoader(subset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)

    model     = SmallResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"],
                          momentum=config["momentum"],
                          weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs_per_trial"])

    wandb.init(
        project=config["wandb_project"],
        name=f"trial-{trial_idx:02d}",
        group="imbalanced-random-experiment",
        tags=["trial"],
        config={**config, "trial": trial_idx},
        reinit=True,
    )

    best_test_acc = 0.0
    for epoch in range(config["epochs_per_trial"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device)
        scheduler.step()
        best_test_acc = max(best_test_acc, test_acc)
        wandb.log({
            "epoch":      epoch + 1,
            "train/loss": train_loss,
            "train/acc":  train_acc,
            "test/loss":  test_loss,
            "test/acc":   test_acc,
            "lr":         scheduler.get_last_lr()[0],
        })

    wandb.summary["final_test_acc"] = test_acc
    wandb.summary["best_test_acc"]  = best_test_acc
    wandb.finish()

    print(f"  Trial {trial_idx:02d}/{config['num_trials']} | "
          f"final={test_acc*100:.2f}%  best={best_test_acc*100:.2f}%")
    return float(best_test_acc)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(config):
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_full, test_set = get_datasets(config)
    cls_num_list = train_full.get_cls_num_list()
    n_train      = len(train_full)

    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)

    print(f"Device       : {device}")
    print(f"Imbalance    : type={config['imb_type']}  factor={config['imb_factor']}")
    print(f"Training set : {n_train} images")
    print(f"Class counts : {cls_num_list}")
    print(f"Trials       : {config['num_trials']}")
    print(f"Subset size  : {config['subset_size']} / {n_train}")
    print(f"Epochs/trial : {config['epochs_per_trial']}\n")

    all_accs = []
    for i in range(1, config["num_trials"] + 1):
        acc = run_trial(i, train_full, test_loader, config, device)
        all_accs.append(acc)

    accs       = np.array(all_accs) * 100
    mean_acc   = accs.mean()
    std_acc    = accs.std()
    min_acc    = accs.min()
    max_acc    = accs.max()
    median_acc = np.median(accs)

    print("\n" + "="*55)
    print("  IMBALANCED RANDOM SUBSET SELECTION RESULTS")
    print("="*55)
    print(f"  Imbalance   : {config['imb_type']}  factor={config['imb_factor']}")
    print(f"  Trials      : {config['num_trials']}")
    print(f"  Subset size : {config['subset_size']} / {n_train}")
    print(f"  Mean acc    : {mean_acc:.2f}%")
    print(f"  Std dev     : {std_acc:.2f}%")
    print(f"  Min acc     : {min_acc:.2f}%")
    print(f"  Max acc     : {max_acc:.2f}%")
    print(f"  Median acc  : {median_acc:.2f}%")
    print("="*55)

    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(accs, bins=15, color="#4C72B0", edgecolor="white",
            linewidth=0.8, alpha=0.85)
    ax.axvline(mean_acc,   color="#DD4444", lw=2, linestyle="--",
               label=f"Mean {mean_acc:.2f}%")
    ax.axvline(median_acc, color="#44AA44", lw=2, linestyle=":",
               label=f"Median {median_acc:.2f}%")
    ax.axvspan(mean_acc - std_acc, mean_acc + std_acc,
               alpha=0.12, color="#DD4444", label=f"±1 std ({std_acc:.2f}%)")
    for val, label in [(min_acc, f"Min\n{min_acc:.1f}%"),
                       (max_acc, f"Max\n{max_acc:.1f}%")]:
        ax.axvline(val, color="#888888", lw=1.5, linestyle="-.")
        ax.text(val, ax.get_ylim()[1] * 0.92, label,
                ha="center", va="top", fontsize=8, color="#555555")
    ax.set_xlabel("Best Test Accuracy (%)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Test Accuracy over {config['num_trials']} Random Subsets\n"
        f"(imb={config['imb_type']} factor={config['imb_factor']}, "
        f"subset={config['subset_size']}/{n_train})",
        fontsize=11,
    )
    ax.legend(fontsize=9)

    # Class distribution bar chart
    ax2 = axes[1]
    ax2.bar(range(10), cls_num_list, color="#55A868", edgecolor="white")
    ax2.set_xlabel("Class", fontsize=12)
    ax2.set_ylabel("Number of samples", fontsize=12)
    ax2.set_title(
        f"Class distribution after imbalancing\n"
        f"(type={config['imb_type']}, factor={config['imb_factor']})",
        fontsize=11,
    )
    ax2.set_xticks(range(10))

    plt.tight_layout()
    hist_path = f"imbalanced_random_hist_{config['imb_type']}_{config['imb_factor']}.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Plot saved -> {hist_path}")

    # W&B summary
    wandb.init(
        project=config["wandb_project"],
        name="experiment-summary",
        group="imbalanced-random-experiment",
        tags=["summary"],
        config=config,
        reinit=True,
    )
    wandb.log({
        "summary/mean_acc":       mean_acc,
        "summary/std_acc":        std_acc,
        "summary/min_acc":        min_acc,
        "summary/max_acc":        max_acc,
        "summary/median_acc":     median_acc,
        "summary/histogram":      wandb.Image(hist_path),
        "summary/all_trial_accs": wandb.Histogram(accs),
    })

    cls_table = wandb.Table(columns=["class", "num_samples"])
    for cls_idx, cnt in enumerate(cls_num_list):
        cls_table.add_data(cls_idx, cnt)
    wandb.log({"summary/class_distribution": cls_table})

    trial_table = wandb.Table(columns=["trial", "best_test_acc_%"])
    for i, a in enumerate(accs, 1):
        trial_table.add_data(i, round(float(a), 4))
    wandb.log({"summary/trial_results": trial_table})
    wandb.finish()

    print("All results logged to W&B. Done!")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Imbalanced Random Subset Experiment")
    parser.add_argument("--subset_size",      type=int,   default=DEFAULT_CONFIG["subset_size"])
    parser.add_argument("--num_trials",       type=int,   default=DEFAULT_CONFIG["num_trials"])
    parser.add_argument("--epochs_per_trial", type=int,   default=DEFAULT_CONFIG["epochs_per_trial"])
    parser.add_argument("--batch_size",       type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",               type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--seed",             type=int,   default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--imb_type",         type=str,   default=DEFAULT_CONFIG["imb_type"],
                        help="exp or step")
    parser.add_argument("--imb_factor",       type=float, default=DEFAULT_CONFIG["imb_factor"],
                        help="ratio of min to max class size")
    parser.add_argument("--wandb_project",    type=str,   default=DEFAULT_CONFIG["wandb_project"])
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG, **vars(args)}
    main(config)