# wts-vision

# Weak-to-Strong Experiments on ImageNette

Imagenette was chosen due to logistic and compute restrictions (bigger experiments coming soon!)

Inspired by OpenAI's research at (https://github.com/openai/weak-to-strong/tree/main)

Original Approach: Generate the weak labels using an [AlexNet](https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html) model pretrained on ImageNette and we use linear probes on top of [DINO](https://github.com/facebookresearch/dino) models as a strong student.

Modified Approach: Use the weak labels to generate pairwise preferences to train a linear probe, eliciting a stronger weak-to-strong supervision.

Set download=True in data.py for the first run

```bash
python3 run_weak_strong.py \
    data_path: <DATA_PATH> \
    weak_model_name: <WEAK_MODEL>\
    strong_model_name: <STRONG_MODEL> \
    batch_size <BATCH_SIZE> \
    seed <SEED> \
    n_epochs <N_EPOCHS> \
    lr <LR> \
    n_train <N_TRAIN>
```

Run the prefs version using the same command. Refer to the OpenAI GitHub for detailed parameter explanations.

With the commands above we get the following results (note that the results may not reproduce exactly due to randomness):

AlexNet (weak label) Accuracy: 0.089

DINO ResNet50 (strong on gt) Accuracy: 0.91

| Model                           | PGR   |
| ------------------------------- | ----- |
| AlexNet → DINO ResNet50         | 0.096 |
| AlexNet → DINO ResNet50 (Prefs) | 0.132 |

Scores are low, but better for prefs. More extensive testing coming soon!

You can add new custom models to the `models.py` and new datasets to `data.py`.
