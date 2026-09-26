# Sports News and Tweet Emotion Classification

Text classification using TensorFlow/Keras neural networks and binary bag-of-words features.

This repository contains two projects:

1. **Tweet Emotion Classification:** Classifies tweets into six emotions and compares a baseline neural network with a deeper model.
2. **BBC Sports News Classification:** Classifies news articles into five sports categories.

## Repository Files

| File | Description |
|---|---|
| [sports-news-code.ipynb](sports-news-code.ipynb) | BBC sports classification notebook |
| [twee-emotion-code.ipynb](twee-emotion-code.ipynb) | Tweet emotion classification notebook |
| [bbcsports.csv](bbcsports.csv) | Sports news dataset |
| [tweet_emotions.csv](tweet_emotions.csv) | Tweet emotion dataset |

## Datasets

### Tweet Emotions

The dataset contains **21,051 tweets**. The model uses the `Tweet` column as input and the `Label` column as its target.

| Emotion | Number of Records |
|---|---:|
| anger | 1,555 |
| disgust | 761 |
| fear | 2,816 |
| joy | 8,240 |
| sadness | 3,830 |
| surprise | 3,849 |

### BBC Sports News

The dataset contains **737 articles**. The model uses the `text` column as input and the `label` column as its target.

| Category | Number of Records |
|---|---:|
| athletics | 101 |
| cricket | 124 |
| football | 265 |
| rugby | 147 |
| tennis | 100 |

## Preprocessing

Both notebooks use the following workflow:

1. Load and inspect the CSV dataset using pandas.
2. Convert class labels into one-hot vectors using `pd.get_dummies`.
3. Create an **80% training / 20% testing** split with stratification and `random_state=42`.
4. Fit a Keras tokenizer on the training partition with a vocabulary limit of **10,000**.
5. Convert text into binary word-presence vectors.
6. Reserve **10% of the training partition** for validation.

Each input vector represents which words occur in a text. This approach does not preserve word order or word frequency.

### Effective Split Sizes

| Dataset | Model Training | Validation | Testing |
|---|---:|---:|---:|
| Tweet emotions | 15,156 | 1,684 | 4,211 |
| BBC sports | 530 | 59 | 148 |

The tokenizer is fitted before the validation split, so its vocabulary includes validation texts but excludes test texts.

## Model Architectures

The models are fully connected neural networks built using `keras.Sequential`.

### Tweet Baseline

```text
Input: 10,000 features
        ↓
Dense: 64 neurons, ReLU
        ↓
Dense: 32 neurons, ReLU
        ↓
Output: 6 neurons, Softmax
```

### Deeper Tweet Model

```text
Input: 10,000 features
        ↓
Dense: 64 neurons, ReLU
        ↓
Dense: 32 neurons, ReLU
        ↓
Dense: 32 neurons, ReLU
        ↓
Dense: 32 neurons, ReLU
        ↓
Output: 6 neurons, Softmax
```

### BBC Sports Model

```text
Input: 10,000 features
        ↓
Dense: 64 neurons, ReLU
        ↓
Dense: 32 neurons, ReLU
        ↓
Output: 5 neurons, Softmax
```

## Training Configuration

| Setting | Value |
|---|---|
| Optimizer | RMSprop |
| Loss | Categorical cross-entropy |
| Metric | Accuracy |
| Batch size | 32 |
| Configured epochs | 20 |

Both tweet models use early stopping:

```python
EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)
```

In the saved runs, both tweet models stopped after **four epochs** and restored the best weights from **epoch two**.

The BBC sports model uses **20 epochs without early stopping**.

## Results

The following results are recorded in the notebooks’ saved outputs.

| Model | Test Accuracy | Test Loss |
|---|---:|---:|
| Tweet baseline | **59.77%** | 1.1251 |
| Deeper tweet model | **57.82%** | 1.1513 |
| BBC sports model | **98.65%** | 0.0500 |

### Effect of Adding More Layers

Adding two hidden layers reduced tweet classification accuracy by **1.95 percentage points** in this run.

The deeper model did not generalize better. Both tweet models showed increasing training accuracy while validation loss worsened after epoch two, supporting the use of early stopping.

### Tweet Emotion Performance

The baseline model’s classification report shows:

| Emotion | Precision | Recall | F1-score |
|---|---:|---:|---:|
| anger | 0.49 | 0.33 | 0.40 |
| disgust | 0.46 | 0.08 | 0.13 |
| fear | 0.70 | 0.51 | 0.59 |
| joy | 0.66 | 0.80 | 0.72 |
| sadness | 0.47 | 0.52 | 0.49 |
| surprise | 0.56 | 0.53 | 0.54 |

- **Macro F1-score:** 0.48
- **Weighted F1-score:** 0.58

Joy has the largest number of examples and the highest recall. Disgust has the fewest examples and the lowest recall. Class imbalance and ambiguous emotional language may contribute to these differences.

### BBC Sports Performance

The model correctly classified **146 of 148 test articles**.

| Category | Test Recall |
|---|---:|
| athletics | 100.00% |
| cricket | 96.00% |
| football | 100.00% |
| rugby | 96.67% |
| tennis | 100.00% |

Distinct topic vocabulary may make sports classification easier than emotion detection. However, the two datasets differ, so their accuracies are not a controlled comparison of model quality.

## How to Run

### Google Colab

1. Download a notebook from this repository and open it in Google Colab.
2. Upload the corresponding CSV to Google Drive.
3. Run the Google Drive mount cell.
4. Update the `pd.read_csv` path to your CSV location.
5. Restart the runtime and run all cells in order.

The original notebooks use these dataset paths:

```text
/content/drive/MyDrive/Colab Notebooks/Datasets/tweet_emotions.csv
/content/drive/MyDrive/Colab Notebooks/Datasets/bbcsports.csv
```

### Local Jupyter Notebook

Clone the repository:

```bash
git clone https://github.com/ishfaque2000/sports-news-and-tweet-emotion-classification.git
cd sports-news-and-tweet-emotion-classification
```

Install the required libraries:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib jupyter
```

Skip the Google Colab Drive-mount cell and update the dataset-loading statement in each notebook:

```python
# Tweet emotion notebook
df = pd.read_csv("tweet_emotions.csv")

# Sports news notebook
df = pd.read_csv("bbcsports.csv")
```

Start Jupyter:

```bash
jupyter notebook
```

Open the desired notebook and run its cells in order.

## Included Analysis

- Dataset exploration and class counts
- Text tokenization and binary vectorization
- Neural network architecture summaries
- Training and validation accuracy/loss plots
- Test accuracy and loss
- Precision, recall, and F1-score reports
- Per-class performance discussion
- Comparison of baseline and deeper tweet models

## Limitations and Reproducibility

- Results reflect saved notebook runs and may vary after retraining.
- The split uses a fixed seed, but the initial tweet model does not set a model seed before construction.
- Binary bag-of-words features lose word order and context.
- Emotion classes are imbalanced, so accuracy should be interpreted alongside per-class recall and macro F1.
- The vocabulary includes validation texts. A stricter evaluation would split validation data before fitting the tokenizer.
- The printed training accuracy includes the validation subset because evaluation uses the entire original training partition.
- The BBC test set contains only 148 articles; its high accuracy should not be assumed to generalize to all sports news.
- The notebooks use the legacy Keras `Tokenizer` API. Library versions are not pinned.

## Technologies

- Python
- TensorFlow / Keras
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Google Colab / Jupyter Notebook

## Dataset Attribution

These datasets were supplied for educational work. Original source links and redistribution terms are not recorded in the notebooks. Dataset ownership remains with the respective owners; this repository does not grant a separate license for the data.
