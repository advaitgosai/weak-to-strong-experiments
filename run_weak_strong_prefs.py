import fire
import numpy as np
import torch
import tqdm
from data import get_imagenet, get_imagenette
from models import alexnet, resnet50_dino, vitb8_dino
from torch import nn


def get_model(name):
    if name == "alexnet":
        model = alexnet()
    elif name == "resnet50_dino":
        model = resnet50_dino()
    elif name == "vitb8_dino":
        model = vitb8_dino()
    else:
        raise ValueError(f"Unknown model {name}")
    model.cuda()
    model.eval()
    model = nn.DataParallel(model)
    return model


def get_embeddings(model, loader):
    all_embeddings, all_y, all_probs = [], [], []

    for x, y in tqdm.tqdm(loader):
        output = model(x.cuda())
        if len(output) == 2:
            embeddings, logits = output
            probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu()
            all_probs.append(probs)
        else:
            embeddings = output

        all_embeddings.append(embeddings.detach().cpu())
        all_y.append(y)

    all_embeddings = torch.cat(all_embeddings, axis=0)
    all_y = torch.cat(all_y, axis=0)
    if len(all_probs) > 0:
        all_probs = torch.cat(all_probs, axis=0)
        acc = (torch.argmax(all_probs, dim=1) == all_y).float().mean()
    else:
        all_probs = None
        acc = None
    return all_embeddings, all_y, all_probs, acc



def train_logreg(
    x_train,
    y_train,
    eval_datasets,
    n_epochs=10,
    weight_decay=0.0,
    lr=1.0e-3,
    batch_size=100,
    n_classes=1000,
):
    x_train = x_train.float()
    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)

    d = x_train.shape[1]
    model = torch.nn.Linear(d, n_classes).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    n_batches = len(train_loader)
    n_iter = n_batches * n_epochs
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_iter)

    results = {f"{key}_all": [] for key in eval_datasets.keys()}
    for epoch in (pbar := tqdm.tqdm(range(n_epochs), desc="Epoch 0")):
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            schedule.step()
            if len(y.shape) > 1:
                y = torch.argmax(y, dim=1)
            correct += (torch.argmax(pred, -1) == y).detach().float().sum().item()
            total += len(y)
        pbar.set_description(f"Epoch {epoch}, Train Acc {correct / total:.3f}")

        for key, (x_test, y_test) in eval_datasets.items():
            x_test = x_test.float().cuda()
            pred = torch.argmax(model(x_test), axis=-1).detach().cpu()
            acc = (pred == y_test).float().mean()
            results[f"{key}_all"].append(acc)

    for key in eval_datasets.keys():
        results[key] = results[f"{key}_all"][-1]
    return results


def generate_preference_pairs(embeddings, weak_labels, rng):
    pairs = []
    preferences = []
    n = len(embeddings)
    for i in tqdm.tqdm(range(n), desc="Generating Preference Pairs"):
        for j in range(i + 1, n):
            if torch.argmax(weak_labels[i]) != torch.argmax(weak_labels[j]):
                continue
            if torch.max(weak_labels[i]) > torch.max(weak_labels[j]):
                pairs.append((embeddings[i] - embeddings[j]).detach().cpu().numpy())
                preferences.append(1)
            else:
                pairs.append((embeddings[j] - embeddings[i]).detach().cpu().numpy())
                preferences.append(0)
    pairs, preferences = np.array(pairs), np.array(preferences)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)
    return pairs[indices], preferences[indices]


def train_logreg_on_pairs(x_train, y_train, eval_datasets, n_epochs=10, lr=1.0e-3, batch_size=100):
    x_train = x_train.float()
    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size)
    model = torch.nn.Linear(x_train.shape[1], 2).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    results = {f"{key}_all": [] for key in eval_datasets.keys()}
    for epoch in tqdm.tqdm(range(n_epochs), desc="Epoch"):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        for key, (x_test, y_test) in eval_datasets.items():
            x_test, y_test = x_test.float().cuda(), y_test.cuda()
            with torch.no_grad():
                outputs = model(x_test)
                predicted = outputs.max(1)[1]
                accuracy = (predicted == y_test).float().mean()
                results[f"{key}_all"].append(accuracy)
    for key in eval_datasets.keys():
        results[key] = results[f"{key}_all"][-1]
    return model, results

# def infer_class_from_preferences(model, test_embeddings, reference_embeddings, reference_labels):
#     votes = torch.zeros((len(test_embeddings), len(torch.unique(reference_labels)))).cuda()
#     for i, test_embedding in enumerate(test_embeddings):
#         for class_idx in torch.unique(reference_labels):
#             class_mask = reference_labels == class_idx
#             class_embeddings = reference_embeddings[class_mask]
#             for ref_embedding in class_embeddings:
#                 pair = torch.tensor(test_embedding - ref_embedding).float().unsqueeze(0).cuda()
#                 output = model(pair)
#                 preference = output.argmax(dim=1).item()
#                 votes[i, class_idx] += preference
#     _, predicted_classes = votes.max(dim=1)
#     return predicted_classes.cpu()

# def infer_class_from_preferences(model, test_embeddings, reference_embeddings, reference_labels):
#     votes = torch.zeros((len(test_embeddings), len(torch.unique(reference_labels)))).cuda()
#     num_refs_per_test = min(len(reference_embeddings), 10)  # Example: 10 reference images

#     for i in tqdm.tqdm(range(len(test_embeddings)), desc="Inferring Classes"):
#         test_embedding = test_embeddings[i]
#         for class_idx in torch.unique(reference_labels):
#             class_mask = reference_labels == class_idx
#             class_embeddings = reference_embeddings[class_mask]
#             for ref_embedding in class_embeddings:
#                 pair = torch.tensor(test_embedding - ref_embedding).float().unsqueeze(0).cuda()
#                 output = model(pair)
#                 preference = output.argmax(dim=1).item()
#                 votes[i, class_idx] += preference

#     _, predicted_classes = votes.max(dim=1)
#     return predicted_classes.cpu()

def infer_class_from_preferences(model, test_embeddings, reference_embeddings, reference_labels, num_refs=10):
    # num_refs: Number of reference embeddings to compare against for each class
    votes = torch.zeros((len(test_embeddings), len(torch.unique(reference_labels)))).cuda()
    for class_idx in tqdm.tqdm(torch.unique(reference_labels)):
        class_mask = reference_labels == class_idx
        class_embeddings = reference_embeddings[class_mask]
        if len(class_embeddings) > num_refs:
            indices = torch.randperm(len(class_embeddings))[:num_refs].cuda()
            class_embeddings = class_embeddings[indices]
        for i, test_embedding in enumerate(test_embeddings):
            diffs = test_embedding - class_embeddings 
            outputs = model(diffs)
            preference_counts = outputs.argmax(dim=1).sum().item()
            votes[i, class_idx] += preference_counts
    _, predicted_classes = votes.max(dim=1)
    return predicted_classes.cpu()


def main(
    batch_size: int = 16,
    weak_model_name: str = "alexnet",
    strong_model_name: str = "resnet50_dino",
    n_train: int = 3000,
    seed: int = 0,
    data_path: str = "datasets/imagenette/",
    n_epochs: int = 10,
    lr: float = 1e-3,
):
    weak_model = get_model(weak_model_name)
    strong_model = get_model(strong_model_name)
    _, loader = get_imagenette(data_path, split="val", batch_size=batch_size, shuffle=False)
    print("Getting weak labels...")
    _, gt_labels, weak_labels, weak_acc = get_embeddings(weak_model, loader)
    print(f"Weak label accuracy: {weak_acc:.3f}")
    print("Getting strong embeddings...")
    embeddings, strong_gt_labels, _, _ = get_embeddings(strong_model, loader)
    assert torch.all(gt_labels == strong_gt_labels)
    del strong_gt_labels
    
    order = np.arange(len(embeddings))
    rng = np.random.default_rng(seed)
    
#     #### NORMAL APPROACH ####
    
    rng.shuffle(order)
    x = embeddings[order]
    y = gt_labels[order]
    yw = weak_labels[order]
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    yw_train, yw_test = yw[:n_train], yw[n_train:]
    yw_test = torch.argmax(yw_test, dim=1)
    eval_datasets = {"test": (x_test, y_test), "test_weak": (x_test, yw_test)}
    
    print("Training logreg on weak labels...")
    results_weak = train_logreg(x_train, yw_train, eval_datasets, n_epochs=n_epochs, lr=lr)
    print(f"Final accuracy: {results_weak['test']:.3f}")
    print(f"Final supervisor-student agreement: {results_weak['test_weak']:.3f}")
    print(f"Accuracy by epoch: {[acc.item() for acc in results_weak['test_all']]}")
    print(
        f"Supervisor-student agreement by epoch: {[acc.item() for acc in results_weak['test_weak_all']]}"
    )

    print("Training logreg on ground truth labels...")
    results_gt = train_logreg(x_train, y_train, eval_datasets, n_epochs=n_epochs, lr=lr)
    print(f"Final accuracy: {results_gt['test']:.3f}")
    print(f"Accuracy by epoch: {[acc.item() for acc in results_gt['test_all']]}")

    print("\n\n" + "=" * 100)
    print(f"Weak label accuracy: {weak_acc:.3f}")
    print(f"Weak→Strong accuracy: {results_weak['test']:.3f}")
    print(f"Strong accuracy: {results_gt['test']:.3f}")
    print(f"PGR = {(results_weak['test'] - weak_acc) / (results_gt['test'] - weak_acc)}")
    print("=" * 100)

    
    print("Training logreg on weak preferences...")
    x_pairs_train, prefs_train = generate_preference_pairs(x_train, yw_train, rng)
    x_pairs_test, prefs_test = generate_preference_pairs(x_test, yw_test, rng)
    x_pairs_train, prefs_train = torch.tensor(x_pairs_train).float().cuda(), torch.tensor(prefs_train).long().cuda()
    x_pairs_test, prefs_test = torch.tensor(x_pairs_test).float().cuda(), torch.tensor(prefs_test).long().cuda()
    eval_datasets_pairs = {"test_pairs": (x_pairs_test, prefs_test)}

    model_pairs, results_weak_pairs = train_logreg_on_pairs(x_pairs_train, prefs_train, eval_datasets_pairs, n_epochs=n_epochs, lr=lr)
    print(f"Final pairwise accuracy: {results_weak_pairs['test_pairs']:.3f}")
    
    # Use the trained model to infer class labels for the test set
    # Infer class labels from pairwise preferences
    predicted_classes = infer_class_from_preferences(model_pairs, x_pairs_test, x_train, y_train)
    test_accuracy = (predicted_classes == y_test.cpu()).float().mean().item()
    print(f"Classification accuracy inferred from pairwise preferences: {test_accuracy:.3f}")

    # Calculate PGR using weak_pairs and ground truth results
    # Note: PGR calculation might need to be adjusted based on how you define success in the pairwise setting
    # weak_acc_pairs = results_weak_pairs['test_pairs']
    ground_truth_acc = results_gt['test']
    # Assuming weak_acc is the accuracy of the weak model on the test set (direct classification, not pairwise)
    pgr = (test_accuracy - weak_acc) / (ground_truth_acc - weak_acc)
    print(f"PGR (based on inferred classification accuracy): {pgr}")
    
    #### PREF BASED APPROACH ####
    
    # Generate preference pairs
    print("\n\n" + "=" * 100)
    x_pairs, y_pairs = generate_preference_pairs(embeddings, weak_labels, rng)
    x_pairs, y_pairs = torch.tensor(x_pairs).float(), torch.tensor(y_pairs).long()

    # Split the data into training and evaluation sets
    x_train, x_eval = x_pairs[:n_train], x_pairs[n_train:]
    y_train, y_eval = y_pairs[:n_train], y_pairs[n_train:]

    # Train logistic regression on preference pairs
    print("Training logistic regression on preference pairs...")
    eval_datasets = {"eval": (x_eval.cuda(), y_eval.cuda())}
    model, results = train_logreg_on_pairs(x_train.cuda(), y_train.cuda(), eval_datasets, n_epochs=n_epochs, lr=lr)
    print(f"Pairwise classification accuracy: {results['eval']:.3f}")

    # Infer class from preferences
    print("Inferring classes from preferences...")
    predicted_classes = infer_class_from_preferences(model, x_eval.cuda(), embeddings[:n_train].cuda(), gt_labels[:n_train].cuda())
    accuracy = (predicted_classes == gt_labels[n_train:].cpu()).float().mean().item()
    print(f"Classification accuracy inferred from preferences: {accuracy:.3f}")

    # Calculate and display the PGR
    pgr = (accuracy - weak_acc) / (1.0 - weak_acc)
    print(f"Preference-based PGR: {pgr:.3f}")

if __name__ == "__main__":
    fire.Fire(main)