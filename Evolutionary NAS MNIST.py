"""
Evolutionary NAS MNIST
"""

import copy
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import trange

# ----------------------------
# Config / Representation
# ----------------------------
DEFAULT_INPUT_CHANNELS = 1
NUM_CLASSES = 10
Arch = list  # list of dicts for blocks

def random_block(min_channels=8, max_channels=64):
    return {
        'out_channels': random.choice([8, 16, 24, 32, 48, 64]),
        'kernel': random.choice([3, 5]),
        'pool': random.choice([False, True])
    }

def random_arch(min_layers=1, max_layers=4):
    return [random_block() for _ in range(random.randint(min_layers, max_layers))]

def arch_to_str(arch):
    return ' | '.join(f"C{b['out_channels']}k{b['kernel']}{'P' if b['pool'] else ''}" for b in arch)

# ----------------------------
# Model builder
# ----------------------------
class SimpleCNN(nn.Module):
    def _init_(self, arch: Arch, in_channels=DEFAULT_INPUT_CHANNELS, num_classes=NUM_CLASSES):
        super()._init_()
        layers = []
        cur_c = in_channels
        for i, block in enumerate(arch):
            out_c = block['out_channels']
            k = block['kernel']
            padding = k // 2
            layers.append(nn.Conv2d(cur_c, out_c, kernel_size=k, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            if block['pool']:
                layers.append(nn.MaxPool2d(2))
            cur_c = out_c
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(cur_c, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ----------------------------
# Weight inheritance helper
# ----------------------------
def try_inherit_weights(child: nn.Module, parent: nn.Module):
    child_dict = child.state_dict()
    parent_dict = parent.state_dict()
    matched = 0
    for k, v in parent_dict.items():
        if k in child_dict and child_dict[k].shape == v.shape:
            child_dict[k] = v.clone()
            matched += 1
    child.load_state_dict(child_dict)
    return matched

# ----------------------------
# Data / training utils
# ----------------------------
def get_dataloaders(batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    val_size = 5000
    train_size = len(trainset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(trainset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, valloader, testloader

def train_one_epoch(model, device, trainloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc

def evaluate(model, device, loader, criterion=None):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if criterion:
                loss = criterion(outputs, targets)
                loss_sum += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = correct / total
    avg_loss = loss_sum / total if criterion else None
    return avg_loss, acc

# ----------------------------
# Evolutionary algorithm
# ----------------------------
Individual = namedtuple('Individual', ['arch', 'fitness', 'model_state'])

def mutate_arch(arch: Arch, max_layers=6):
    new = copy.deepcopy(arch)
    ops = ['add', 'remove', 'modify']
    op = random.choice(ops)
    if op == 'add' and len(new) < max_layers:
        pos = random.randint(0, len(new))
        new.insert(pos, random_block())
    elif op == 'remove' and len(new) > 1:
        pos = random.randrange(len(new))
        new.pop(pos)
    else:
        pos = random.randrange(len(new))
        field = random.choice(['out_channels', 'kernel', 'pool'])
        if field == 'out_channels':
            new[pos]['out_channels'] = random.choice([8, 16, 24, 32, 48, 64])
        elif field == 'kernel':
            new[pos]['kernel'] = random.choice([3, 5])
        else:
            new[pos]['pool'] = not new[pos]['pool']
    if random.random() < 0.05 and len(new) >= 2:
        i, j = random.sample(range(len(new)), 2)
        new[i], new[j] = new[j], new[i]
    return new

def evolve(population, trainloader, valloader, device, args):
    population = sorted(population, key=lambda x: x.fitness if x.fitness is not None else 0.0, reverse=True)
    next_pop = []
    K = max(1, int(args.elitism * len(population)))
    next_pop.extend(population[:K])

    while len(next_pop) < args.pop_size:
        tournament = random.sample(population, k=min(args.tournament_k, len(population)))
        parent = max(tournament, key=lambda x: x.fitness if x.fitness is not None else 0.0)
        child_arch = mutate_arch(parent.arch, max_layers=args.max_layers)

        child_model = SimpleCNN(child_arch).to(device)
        parent_model = SimpleCNN(parent.arch).to(device)
        if parent.model_state:
            parent_model.load_state_dict(parent.model_state)
        if args.inherit_weights:
            try_inherit_weights(child_model, parent_model)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(child_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        for ep in range(args.train_epochs):
            train_one_epoch(child_model, device, trainloader, optimizer, criterion)
        _, val_acc = evaluate(child_model, device, valloader, None)
        child_state = child_model.state_dict()
        child = Individual(arch=child_arch, fitness=val_acc, model_state=child_state)
        next_pop.append(child)
    return next_pop

# ----------------------------
# Main Evolution Run
# ----------------------------
def run_evolution(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    trainloader, valloader, testloader = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)

    # init population
    population = []
    print("Initializing population...")
    for i in range(args.pop_size):
        arch = random_arch(min_layers=args.min_layers, max_layers=args.init_max_layers)
        model = SimpleCNN(arch).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        for ep in range(args.init_train_epochs):
            train_one_epoch(model, device, trainloader, optimizer, criterion)
        _, val_acc = evaluate(model, device, valloader, None)
        state = model.state_dict()
        population.append(Individual(arch=arch, fitness=val_acc, model_state=state))
        print(f" Init {i+1}/{args.pop_size}: {arch_to_str(arch)}  val_acc={val_acc:.4f}")

    best = None
    for gen in range(1, args.generations + 1):
        print(f"\n=== Generation {gen} ===")
        population = evolve(population, trainloader, valloader, device, args)
        population = sorted(population, key=lambda x: x.fitness if x.fitness is not None else 0.0, reverse=True)
        best = population[0]
        print(f" Best gen {gen}: {arch_to_str(best.arch)}  val_acc={best.fitness:.4f}")

    print("\n=== Final Best ===")
    best_model = SimpleCNN(best.arch).to(device)
    best_model.load_state_dict(best.model_state)
    _, test_acc = evaluate(best_model, device, testloader, None)
    print(f" Best arch: {arch_to_str(best.arch)} | val={best.fitness:.4f} | test_acc={test_acc:.4f}")
    return best

# ----------------------------
# Fixed Arguments for Colab
# ----------------------------
class Args:
    pop_size = 4
    generations = 3
    train_epochs = 1
    init_train_epochs = 1
    batch_size = 128
    lr = 0.05
    elitism = 0.2
    tournament_k = 2
    min_layers = 1
    init_max_layers = 3
    max_layers = 6
    inherit_weights = False
    num_workers = 2

args = Args()
run_evolution(args)
