# Diagnostic information preservation in qubit-limited quantum medical image classification

## Abstract

Small quantum models need medical images compressed into a few numbers, which may erase useful information before training. This study reduced three MedMNIST datasets—chest X-rays, breast ultrasounds, and colorectal tissue images—to 4–64 numbers using principal component analysis (PCA), a method that summarizes pixel patterns. Full-image and compressed models were compared after regions were covered, details blurred, contrast lowered, or noise added. The study measured how each edit changed prediction confidence. Accuracy and the pixel variation kept did not reliably reveal what survived compression. With four numbers, PneumoniaMNIST reached 84.4% balanced accuracy (equal weight for both classes), slightly above 80.0% using all pixels. Yet covering the central 8×8 pixels (8.2% of the image) changed confidence by only 1.3 percentage points on average, compared with 15.5 points for the full-image model. Colon images kept 62.2% of pixel variation, but texture and edge edits changed confidence only 1% as much as in the full-image model. The quantum model scored 69.2%, below 85.8% for a standard model given the same four numbers. Researchers must therefore separate information lost during compression from the quantum model's limits. This early study does not show that quantum computing performs better and is not intended for clinical use.

## Research question

The project asks:

**When a medical image is reduced to the small number of inputs accepted by a quantum model, which changes in its predicted probabilities survive that reduction?**

This is different from asking only whether classification accuracy goes up or down. A compressed image may still lead to the correct class label even if some of the image details that influenced the full-image model no longer influence the compressed model.

## Terms used in this project

| Term | Meaning in this study |
|---|---|
| Pixel | One value in a digital image. A 28×28 grayscale image has 784 pixels. |
| Model | A computer program that learns patterns from labeled examples and predicts a label for a new image. |
| Classifier | A model that assigns an image to a category, such as “normal” or “pneumonia.” |
| Feature | A number given to a model as input. A raw image uses pixel values as features. |
| Compression | Replacing many pixel values with a smaller number of features. |
| PCA | Principal component analysis. PCA creates new features that capture large patterns of variation among the original pixels. |
| Qubit | The basic unit of information used by a quantum computer. |
| Quantum circuit | A sequence of operations applied to qubits. In this study, the operations contain adjustable values that are learned during training. |
| Classical model | A standard non-quantum machine-learning model run on a conventional computer. |
| Quantum advantage | Evidence that a quantum method outperforms an appropriate non-quantum method on the same task under a fair comparison. |
| Training set | Images and labels used by a model to learn. |
| Test set | Separate images used only to evaluate a trained model. |

## Data

The study uses **MedMNIST**, a public collection of small, standardized biomedical images intended for machine-learning research. The images are 28×28 pixels.

| Dataset | Image type | Classification task | Training images used | Test images used |
|---|---|---|---:|---:|
| PneumoniaMNIST | Pediatric chest X-rays | Normal or pneumonia | 720 | 270 |
| BreastMNIST | Breast ultrasound images | Malignant or normal/benign | 294 | 84 |
| PathMNIST | Microscopic images of colon tissue | Nine tissue categories | 720 | 270 |

The notebook uses balanced working subsets. This means that it selects the same number of examples from each class when possible. Balancing prevents a model from receiving a high score simply by favoring the most common class. These subsets are smaller than the full datasets so the notebook can run on a laptop.

## How the experiment works

### 1. Create a full-image comparison model

A logistic regression model is trained on all pixel values. Logistic regression is a standard statistical classifier that estimates the probability that an example belongs to each class. This model is called the **full-image reference model** because later results are compared with its responses.

The reference model is not assumed to be medically correct. It is simply a fixed comparison point that had access to the full image.

### 2. Reduce each image with PCA

PCA is fitted using only the training images. It identifies combinations of pixels that account for large differences among those images. Each combination becomes one new input number, called a PCA component.

Separate image versions are created with 4, 8, 16, 32, and 64 PCA components. The four-component version is especially important because it can be entered into the four-qubit model used later.

The percentage of **variance retained** records how much of the overall variation among the original pixel values is represented by the selected PCA components. High retained variance means that the compressed data describes much of the dataset’s overall pixel variation. It does not guarantee that every detail affecting a prediction was kept.

### 3. Train models on the compressed images

Several standard models are trained on each PCA version:

- Logistic regression estimates class probabilities using a weighted combination of the input features.
- A random forest combines many decision trees, which are sets of simple if-then rules.
- A multilayer perceptron is a small neural network made of layers of weighted calculations.
- A linear support-vector machine uses the input numbers to find a boundary between classes.
- A dummy classifier ignores the image and predicts from the class frequencies. It provides a basic comparison score.

Using several models helps determine whether a result comes from the compressed data or from the behavior of one particular classification method.

### 4. Make controlled changes to the test images

Each test image is edited in several known ways:

| Image change | What is changed |
|---|---|
| Center occlusion | A square in the center is covered. |
| Off-center occlusion | A square away from the center is covered. |
| Texture weakening | Small local pixel differences are reduced. |
| Boundary weakening | Edges are softened by averaging neighboring pixels. |
| Contrast attenuation | Bright and dark areas are moved closer to the image’s average brightness. |
| Gaussian-like noise | Small random changes are added to the pixels. |
| Patch sweep | A covered square is moved across the image to measure location-specific changes. |

These image changes are experimental tests. They are not intended to reproduce a disease, and they are not statements about which parts of an image are clinically important.

### 5. Compare prediction changes

For each original image, the model produces probabilities for the possible classes. For example, a binary model might assign a 70% probability to pneumonia and a 30% probability to normal. The edited image is then evaluated, and the change in probability is recorded.

The same calculation is performed for the full-image and compressed models:

```text
Original and edited image
          │
          ├── all pixels ── full-image model ── change in predicted probability
          │
          └── PCA numbers ─ compressed model ── change in predicted probability
                                                        │
                                  compare the two changes
```

The **preservation score** summarizes this comparison:

- A score near 1 means that the two models’ probabilities changed by similar amounts.
- A score near 0 means that the compressed model’s response was much smaller or much larger.

The **response ratio** shows the direction of the difference. A ratio below 1 means that the compressed response was weaker. A ratio above 1 means that it was stronger.

Neither measurement proves that the model used medically correct information. They only measure whether the two models reacted similarly.

## Measures reported

| Measure | What it answers |
|---|---|
| Accuracy | What fraction of images received the correct label? |
| Balanced accuracy | What is the average accuracy across classes when every class receives equal weight? |
| Macro F1 | How well did the model balance correct detections with false detections, averaged equally across classes? |
| AUC | How well did the model rank images from less likely to more likely for the correct class across possible decision cutoffs? |
| Variance retained | How much of the overall pixel variation was represented by PCA? |
| Probability shift | How much did the predicted probability change after an image edit? |
| Flip rate | How often did an image edit change the predicted class? |
| Preservation score | How closely did the compressed model’s probability change match the full-image model’s change? |
| Response ratio | Was the compressed model’s response weaker or stronger than the full-image response? |

No single measure answers every question. Accuracy evaluates final labels, while the preservation measures evaluate changes in model response.

## Main results

The table below shows the logistic regression results when each image was reduced to four PCA components.

| Dataset | Pixel variation retained | Balanced accuracy | Average preservation score |
|---|---:|---:|---:|
| PneumoniaMNIST | 58.2% | 84.4% | 47.9% |
| BreastMNIST | 59.9% | 75.0% | 43.0% |
| PathMNIST | 62.2% | 47.8% | 35.8% |

The three datasets retained similar percentages of overall pixel variation, but they did not preserve the same model responses.

For PathMNIST, the four-component model still responded to some covered regions. However, its responses to texture weakening and boundary weakening were about 1% of the full-image responses. For PneumoniaMNIST, the center-occlusion response fell to 8.4% of the full-image response, while the response to reduced contrast increased to 123% of the full-image response.

Compression therefore did not remove every type of image information equally. It weakened some responses, preserved others, and amplified others.

## Quantum-model comparison

The quantum part of the study uses a **variational quantum classifier**. “Variational” means that the circuit contains adjustable values. Training changes those values to reduce classification error. The circuit uses four qubits, so it receives the four PCA components as its input.

The circuit is evaluated with a quantum simulator. A simulator is a program running on a normal computer that calculates how an ideal quantum circuit would behave. Therefore, these results do not include errors or noise from real quantum hardware.

For a fair comparison, the quantum and classical models use the same four PCA numbers, the same 160 training images, and the same 120 test images.

| Model | Balanced accuracy | AUC |
|---|---:|---:|
| Four-qubit quantum classifier | 69.2% | 80.8% |
| Logistic regression | 85.8% | 92.2% |
| Small neural network | 85.8% | 92.3% |
| Dummy classifier | 50.0% | 50.0% |

The quantum model did not outperform the classical models given the same inputs.

## Reliability checks

A result from one small test set may change if different examples are selected. Two checks were used to examine that uncertainty.

**Bootstrap confidence intervals:** The test results were recalculated many times using samples drawn from the original test set. The middle 95% of those recalculated results forms a 95% confidence interval. The quantum model’s balanced-accuracy interval was 61.5%–76.9%. The logistic regression interval was 80.1%–91.4%.

**Shuffled-label negative control:** The training labels were randomly mixed 50 times before logistic regression was trained. A model should not learn a meaningful relationship between images and randomly assigned labels. Ninety-five percent of these random-label scores were at or below 79.6% balanced accuracy. That upper threshold was still higher than the quantum model’s 69.2%. This indicates that the small quantum comparison requires more testing before it can support a strong conclusion.

## Interpretation

The study does not argue that it is surprising for compression to remove information. Instead, it measures where that change appears in this specific experiment.

If the full-image and compressed models respond differently before the quantum model is considered, then at least part of the difference comes from the compressed input. If their responses are similar but the quantum model performs poorly, the model itself becomes a more likely source of the problem.

The results show that classification performance and retained pixel variation do not fully describe what survives compression. Reporting the prediction changes caused by controlled image edits provides additional information about what reached the small model.

## Limitations

This is an undergraduate study designed to run on a laptop. It uses small, balanced subsets rather than every available image. PCA can describe only linear combinations of pixels. The image changes are created by code and are not based on annotations from medical experts. The full-image logistic regression model is not a clinical explanation system. The quantum model was trained once on a simulator rather than repeatedly on real quantum hardware.

Future studies should use complete datasets, repeat quantum training from several starting points, compare additional compression methods, include medically defined image features or regions, test data from other sources, and evaluate real quantum hardware.

## Relation to earlier work

[Singh, Jin, and Merz (2026)](https://doi.org/10.1038/s41598-026-35605-3) tested MedMNIST classification on real IBM quantum hardware after reducing the image data. The present study focuses on the earlier step: determining what changes when an image is reduced before quantum training.

The complete executed analysis is available in [`MedicalMSINTstuff.ipynb`](MedicalMSINTstuff.ipynb).

## Reproducing the analysis

Python 3.9–3.12 is recommended because the MedMNIST software requires PyTorch, a machine-learning library used to load the datasets.

```bash
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
JUPYTER_DATA_DIR="$PWD/.jupyter/share/jupyter" IPYTHONDIR=/tmp/ipython MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl python scripts/execute_revised_notebook.py
```

The first run downloads the selected datasets. The default sample sizes and number of quantum-training rounds are set for laptop execution and can be changed in the notebook.

| Location | Contents |
|---|---|
| [`output/results/`](output/results/) | Numerical results and reliability checks |
| [`output/figures/`](output/figures/) | Graphs and example images |
| [`output/data/raw/`](output/data/raw/) | Downloaded MedMNIST datasets |
| [`output/notebooks/`](output/notebooks/) | Executed notebook copy |
| [`output/logs/`](output/logs/) | Execution records |
| [`output/manifest.json`](output/manifest.json) | List of generated files |
