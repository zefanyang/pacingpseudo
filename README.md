![logo](./images/logo.png)

This repo contains source code for the paper: **Non-iterative Scribble-Supervised Learning for Medical Image Segmentation.** October, 2022. [[arXiv]](https://arxiv.org/pdf/2210.10956.pdf)

### Data information

| Dataset                                                      | Image Modality          | Target Antomy                               | Scribble                                                     | #Patients | #Images | Median Spacing          | Center-Crop Size |
| ------------------------------------------------------------ | ----------------------- | ------------------------------------------- | ------------------------------------------------------------ | --------- | ------- | ----------------------- | ---------------- |
| **CHAOS** [[challenge website]](https://chaos.grand-challenge.org/) | MRI T1-DUAL and T2-SPIR | Liver, Left-kidney, Right-kidney, Spleen    | Manual [[download link]](https://drive.google.com/file/d/1LFfso17fxPaCcwcQJ4lzyKKG22EKGnlt/view?usp=share_link) | 20        | 1917    | 1.62$\times$1.62 mm$^2$ | 256$\times$256   |
| **ACDC** [[challenge website]](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) | Cine-MRI                | Right ventricle, Myocardium, Left ventricle | Manual [[download link]](https://vios-s.github.io/multiscale-adversarial-attention-gates/data) | 100       | 1902    | 1.51$\times$1.51 mm$^2$ | 224$\times$224   |
| **LVSC** [[challenge website]](https://www.cardiacatlas.org/challenges/lv-segmentation-challenge/) | MRI '2D+time'           | Myocardium                                  | Artificial                                                   | 100       | 29,086  | 1.48$\times$1.48 mm$^2$ | 256$\times$256   |

### Data loader

### Model

### Training