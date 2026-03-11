"""
CIFAR-10 Subset Selection Experiment
- Randomly samples a subset of the training set 50 times
- Trains a CNN and evaluates on the full test set each time
- Reports mean, min, max, std of test accuracy + histogram
- Tracks all runs with W&B
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import wandb
import random
import argparse

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEFAULT_CONFIG = dict(
    subset_size=5000,       # number of training samples per trial
    num_trials=50,
    epochs_per_trial=100,    # Updated to 100 epochs
    batch_size=128,
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    seed=42,
    wandb_project="cifar10-subset-selection",
)


# ─────────────────────────────────────────────
# Model: small ResNet-style CNN
# ─────────────────────────────────────────────
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
        self.layer1 = ConvBlock(32, 64, stride=2)
        self.layer2 = ConvBlock(64, 128, stride=2)
        self.layer3 = ConvBlock(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
def get_datasets():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_full = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform)
    return train_full, test_set


# ─────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# Single trial
# ─────────────────────────────────────────────
def run_trial(trial_idx, train_full, test_loader, config, device, parent_run_id):
    # Sample a random subset
    indices = random.sample(range(len(train_full)), config["subset_size"])
    subset = Subset(train_full, indices)
    train_loader = DataLoader(
        subset, batch_size=config["batch_size"], shuffle=True,
        num_workers=2, pin_memory=True)

    model = SmallResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"],
        momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs_per_trial"])

    # Child W&B run for this trial
    run = wandb.init(
        project=config["wandb_project"],
        name=f"trial-{trial_idx:02d}",
        group="subset-experiment",
        tags=["trial"],
        config={**config, "trial": trial_idx, "subset_indices_hash": hash(tuple(sorted(indices)))},
        reinit=True,
    )

    best_test_acc = 0.0
    for epoch in range(config["epochs_per_trial"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        # Track the best accuracy seen so far
        best_test_acc = max(best_test_acc, test_acc)

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "test/loss": test_loss,
            "test/acc": test_acc,
            "best_test_acc": best_test_acc, # Helpful to see it climb in W&B
            "lr": scheduler.get_last_lr()[0],
        })

    # Record the best accuracy to W&B summary and finish run
    wandb.summary["best_test_acc"] = best_test_acc
    wandb.finish()

    print(f"  Trial {trial_idx:02d}/{config['num_trials']} | "
          f"Best test acc: {best_test_acc*100:.2f}%")
          
    # Return the best accuracy instead of the final accuracy
    return best_test_acc


# ─────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────
def main(config):
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Running {config['num_trials']} trials, "
          f"subset size={config['subset_size']}, "
          f"epochs/trial={config['epochs_per_trial']}\n")

    train_full, test_set = get_datasets()
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # Parent summary run
    summary_run = wandb.init(
        project=config["wandb_project"],
        name="experiment-summary",
        group="subset-experiment",
        tags=["summary"],
        config=config,
        reinit=True,
    )
    parent_run_id = summary_run.id
    summary_run.finish()

    # Run all trials
    all_accs = []
    for i in range(1, config["num_trials"] + 1):
        acc = run_trial(i, train_full, test_loader, config, device, parent_run_id)
        all_accs.append(acc)

    # ── Statistics ──────────────────────────────
    accs = np.array(all_accs) * 100  # convert to %
    mean_acc  = accs.mean()
    std_acc   = accs.std()
    min_acc   = accs.min()
    max_acc   = accs.max()
    median_acc = np.median(accs)

    print("\n" + "="*50)
    print("  SUBSET SELECTION EXPERIMENT RESULTS")
    print("="*50)
    print(f"  Trials      : {config['num_trials']}")
    print(f"  Subset size : {config['subset_size']} / {len(train_full)}")
    print(f"  Mean acc    : {mean_acc:.2f}%")
    print(f"  Std dev     : {std_acc:.2f}%")
    print(f"  Min acc     : {min_acc:.2f}%")
    print(f"  Max acc     : {max_acc:.2f}%")
    print(f"  Median acc  : {median_acc:.2f}%")
    print("="*50)

    # ── Histogram ───────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    n, bins, patches = ax.hist(accs, bins=15, color="#4C72B0", edgecolor="white",
                                linewidth=0.8, alpha=0.85)

    # Annotate stats
    ax.axvline(mean_acc,   color="#DD4444", lw=2, linestyle="--", label=f"Mean {mean_acc:.2f}%")
    ax.axvline(median_acc, color="#44AA44", lw=2, linestyle=":",  label=f"Median {median_acc:.2f}%")
    ax.axvspan(mean_acc - std_acc, mean_acc + std_acc,
               alpha=0.12, color="#DD4444", label=f"±1 std ({std_acc:.2f}%)")

    # Min / max ticks
    for val, label in [(min_acc, f"Min\n{min_acc:.1f}%"), (max_acc, f"Max\n{max_acc:.1f}%")]:
        ax.axvline(val, color="#888888", lw=1.5, linestyle="-.")
        ax.text(val, ax.get_ylim()[1] * 0.92, label,
                ha="center", va="top", fontsize=8, color="#555555")

    ax.set_xlabel("Best Test Accuracy (%)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Best Test Accuracy Distribution over {config['num_trials']} Random Subsets\n"
        f"(subset size = {config['subset_size']} / {len(train_full)}, "
        f"{config['epochs_per_trial']} epochs each)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    hist_path = "subset_accuracy_histogram.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"\nHistogram saved → {hist_path}")

    # ── Log summary to W&B ───────────────────────
    summary_run2 = wandb.init(
        project=config["wandb_project"],
        name="experiment-summary",
        group="subset-experiment",
        tags=["summary"],
        config=config,
        reinit=True,
        resume="allow",
        id=parent_run_id,
    )
    wandb.log({
        "summary/mean_acc":   mean_acc,
        "summary/std_acc":    std_acc,
        "summary/min_acc":    min_acc,
        "summary/max_acc":    max_acc,
        "summary/median_acc": median_acc,
        "summary/histogram":  wandb.Image(hist_path),
        "summary/all_trial_best_accs": wandb.Histogram(accs),
    })
    # Log a table of per-trial results
    table = wandb.Table(columns=["trial", "best_test_acc_%"])
    for i, a in enumerate(accs, 1):
        table.add_data(i, round(float(a), 4))
    wandb.log({"summary/trial_results": table})
    wandb.finish()

    print("\nAll results logged to W&B. Done!")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 Subset Selection Experiment")
    parser.add_argument("--subset_size",      type=int,   default=DEFAULT_CONFIG["subset_size"])
    parser.add_argument("--num_trials",       type=int,   default=DEFAULT_CONFIG["num_trials"])
    parser.add_argument("--epochs_per_trial", type=int,   default=DEFAULT_CONFIG["epochs_per_trial"])
    parser.add_argument("--batch_size",       type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",               type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--seed",             type=int,   default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--wandb_project",    type=str,   default=DEFAULT_CONFIG["wandb_project"])
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG, **vars(args)}
    main(config)