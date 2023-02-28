![logo](./images/logo.png)

This repo contains source code for the paper: **Non-iterative Scribble-Supervised Learning for Medical Image Segmentation.** October, 2022. [[arXiv]](https://arxiv.org/pdf/2210.10956.pdf)

### Data information

| Dataset                                                      | Image Modality          | Target Antomy                               | Scribble                                                     | #Images | Median Spacing          | Center-Crop Size |
| ------------------------------------------------------------ | ----------------------- | ------------------------------------------- | ------------------------------------------------------------ | ------- | ----------------------- | ---------------- |
| **CHAOS** [[challenge website]](https://chaos.grand-challenge.org/) | MRI T1-DUAL and T2-SPIR | Liver, Left-kidney, Right-kidney, Spleen    | Manual [[download link]](https://drive.google.com/file/d/1LFfso17fxPaCcwcQJ4lzyKKG22EKGnlt/view?usp=share_link) | 1,917   | 1.62$\times$1.62 mm$^2$ | 256$\times$256   |
| **ACDC** [[challenge website]](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) | Cine-MRI                | Right ventricle, Myocardium, Left ventricle | Manual [[download link]](https://vios-s.github.io/multiscale-adversarial-attention-gates/data) | 1,902   | 1.51$\times$1.51 mm$^2$ | 224$\times$224   |
| **LVSC** [[challenge website]](https://www.cardiacatlas.org/challenges/lv-segmentation-challenge/) | MRI '2D+time'           | Myocardium                                  | Artificial                                                   | 29,086  | 1.48$\times$1.48 mm$^2$ | 256$\times$256   |

### Data

We use five-fold validation to evaluate performance. Images along with five-fold validation text files are stored in **./data/chaos/train_test_split/data_2d** and **./data/chaos/train_test_split/five_fold_split** respectively. 

The dataloaders for augmenting and loading images for network learning is stored in **./datasets/chaos/chaos_dataset.py**.

### Model

We train three models for comparison. They are the baseline model, our pacingpseudo model, and the fully-supervised model. The baseline is the UNet backbone penalized by the partial cross entropy loss. Our pacingpseudo uses the siamese architecture to implement the consistency training. Details are described in the paper [[arXiv]](https://arxiv.org/pdf/2210.10956.pdf). The fullly-supervised model is the UNet trained with the cross-entropy loss and Dice loss.

### Training