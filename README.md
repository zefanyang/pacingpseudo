![logo](./images/logo.png)

This repo contains source code for the scribble-supervised learning paper: **Non-iterative Scribble-Supervised Learning for Medical Image Segmentation.** October, 2022. [[arXiv]](https://arxiv.org/pdf/2210.10956.pdf)

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

### Training and Inference

**train_chaos.py** contains program for traning the baseline model and pacingpseudo model. **upperbound_chaos.py** is for training the fully-supervised model. At the end of each training epoch, we compute Dice similarity coefficients on the validation dataset and plot them using TensorBoard.

**Inference.py** is designed to compute evaluation metrics given a model parameter checkpoint. We use the final checkpoint in our experiments. The metrics includes the Dice similarity coefficients (DSC) and the 95-*th* percentile of Hausdorff distance (HD95). Roughly speaking, DSC quantifies relative overlap and HD95 quantifies boundary distance. Here is a lecture on [Hausdorff distance between convex polygons](http://cgm.cs.mcgill.ca/~godfried/teaching/cg-projects/98/normand/main.html).


### Results

We summarize results in DSC and HD95. DSC is measured in percentages (%) and HD95 is measured in millimeters (mm). Higher is better for DSC and lower is better for HD95.

Below are the results of five-fold validation. Specifically,  in each fold, we first compute the average score of each anatomy over patients, which results in $K$ values. $K$ is the number of anatomies. Then we compute the average of these $K$ values. Since we have 5 folds, we have 5 average values of this kind. The overall average calculate the average value over 5 folds.

Displayed below are experimental results on CHAOS T1.

<center><b>DSC results on CHAOS T1</b></center>

| Models              | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Overall Average |
| ------------------- | ------ | ------ | ------ | ------ | ------ | --------------- |
| Baseline            |        |        |        |        |        |                 |
| PacingPseudo (ours) |        |        |        |        |        |                 |
| Fully-supervised    |        |        |        |        |        |                 |

<center><b>HD95 results on CHAOS T1</b></center>

| Models              | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Overall Average |
| ------------------- | ------ | ------ | ------ | ------ | ------ | --------------- |
| Baseline            |        |        |        |        |        |                 |
| PacingPseudo (ours) |        |        |        |        |        |                 |
| Fully-supervised    |        |        |        |        |        |                 |

### Citation

```latex
@article{yang2022non,
  title={Non-Iterative Scribble-Supervised Learning with Pacing Pseudo-Masks for Medical Image Segmentation},
  author={Yang, Zefan and Lin, Di and Ni, Dong and Wang, Yi},
  journal={arXiv preprint arXiv:2210.10956},
  year={2022}
}
```
