## Weak-to-Strong Generalization with Preferences

### Vision

Imagenette was chosen due to logistic and compute restrictions (bigger experiments coming soon!)

Inspired by OpenAI's research at (https://github.com/openai/weak-to-strong/tree/main)

Original Approach: Generate the weak labels using an [AlexNet](https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html) model pretrained on ImageNette and we use linear probes on top of [DINO](https://github.com/facebookresearch/dino) models as a strong student.

**Modified Approach**: Use the **weak labels to generate pairwise preferences** to train a linear probe, eliciting a stronger weak-to-strong supervision.

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

Scores are low, but better for prefs.

You can add new custom models to `models.py` and new datasets to `data.py`.


### Text

Submitted as a classroom assignment for CS 690S, here are some results:

<img width="688" alt="w2s1" src="https://github.com/advaitgosai/weak-to-strong-experiments/assets/66398068/86746ce3-d123-4991-9583-1bfef91a5cb9">

<img width="688" alt="w2s2" src="https://github.com/advaitgosai/weak-to-strong-experiments/assets/66398068/85ef72d0-95d9-4f31-88cf-874c14572880">

<img width="688" alt="w2s3" src="https://github.com/advaitgosai/weak-to-strong-experiments/assets/66398068/9aa7ff00-e526-4496-aea2-c1fb33cb6261">

<img width="688" alt="w2s4" src="https://github.com/advaitgosai/weak-to-strong-experiments/assets/66398068/5c661ed8-b9e6-40d1-988f-9326e9d2fb64">

#### References:

Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, and Jeff Wu. Weak-to- strong generalization: Eliciting strong capabilities with weak supervision, 2023.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2023.
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017.

Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. Teaching machines to read and comprehend. Advances in neural information processing systems, 28, 2015.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.





