# Star Rating Prediction and Aspect-Based Sentiment Analysis of Restaurant Reviews

ITM 454 Natural Language Processing (Section 001), Final Project, Fall 2026
Instructor: Prof. Monyrath BUNTOUN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Visothh/itm454-restaurant-reviews/blob/main/ITM454%20Star%20Rating%20v6.ipynb)

## What this project does

There are too many restaurant reviews online for anyone to read them all. The star rating gives a quick summary, but it does not tell you *why* the customer liked or did not like the place. So our project does two things with the text of a review:

- **Task A, star rating:** guess the 1 to 5 stars the reviewer gave, only from the text (Yelp reviews).
- **Task B, aspect-based sentiment:** say if the review talks about **Food Quality**, **Service** and **Atmosphere**, and if each one is Positive, Negative, Mixed or Not mentioned (trained on SemEval restaurant sentences). This also works for indirect mentions like "we waited forever", which is about Service even though the word "service" is not there.

The two tasks are trained on different data and are only put together in the demo, so a mistake in one task does not affect the other.

At the end there is a small **Gradio web demo**: you paste a review and it shows the predicted stars and a table for the three aspects.

## Files in this repo

| File | What it is |
|---|---|
| `ITM454 Star Rating v6.ipynb` | **The final notebook.** All the code, step by step, with the outputs of our last run. |
| `report/ITM454 Final Report.docx` | Our final report with all the results and explanations. |
| `labeling/ITM454 Labeling Tool.html` | The small offline tool we made to label reviews by hand (open it in any browser). |
| `results/*_labeled.csv` | Our hand labels: human blind test (50 reviews), manual check of Task B (100 reviews), error causes (50 errors). |
| `requirements.txt` | The Python libraries we used. |

The notebook is one file on purpose. It is split into parts (Part 0 to Part 13, then an Appendix A to H), and each part has a short explanation before the code.

| Part | What it does |
|---|---|
| 0 | Settings, libraries, helper functions |
| 1 | Load the Yelp reviews |
| 2 | Look at the data and clean it |
| 3 | Train / validation / test split by restaurant |
| 4 | Text preprocessing (negation tagging) |
| 5 | Naive Bayes, Logistic Regression, Linear SVM (5-fold cross-validation) |
| 6 | DistilBERT for star rating |
| 7 | Compare all star-rating models |
| 8 | Task B data (SemEval) |
| 9 | Task B step 1: which aspects does a sentence talk about? |
| 10 | Task B step 2: positive or negative about that aspect? |
| 11 | Our own manual checks: blind test, 100 labeled reviews, error causes |
| 12 | Save the models and the Gradio demo |
| 13 | Final tables and figures |
| Appendix | Extra experiments: progress-update model, raw vs clean data, preprocessing test, random vs restaurant split, learning curve, negation and sarcasm, old Task B model, examples |

## Data (not in this repo)

The datasets are too big for GitHub, and the Yelp terms do not allow sharing the data again, so you have to download them yourself.

1. **Yelp Open Dataset (Task A).** Go to https://business.yelp.com/data/resources/open-dataset/, accept the terms and download the JSON version. You get a file called `yelp_dataset.tar` (about 4.3 GB). The notebook reads `business.json` and `review.json` straight from the tar file, so you do not need to unpack it.
   - From the 6,990,280 reviews, we kept the restaurant reviews (4,724,471) and took a random 10% of them (472,292). After removing duplicates and cleaning we had 472,121 reviews.
2. **SemEval-2014 Task 4 and SemEval-2016 Task 5, restaurant sentences (Task B).** You do not need to do anything. The notebook downloads them by itself from https://github.com/davidsbatista/Aspect-Based-Sentiment-Analysis the first time it runs.

## How to run it (Google Colab)

1. In your Google Drive, make a folder called `itm454` (so the path is `MyDrive/itm454`).
2. Upload `yelp_dataset.tar` into that folder.
3. If you want to rerun our hand-label checks (Part 11), make a folder `MyDrive/itm454/results` and put the three `*_labeled.csv` files from this repo in it.
4. Click the **Open in Colab** button at the top of this page (or upload the notebook to Colab yourself).
5. In Colab, go to **Runtime > Change runtime type** and pick a **T4 GPU**. DistilBERT needs a GPU to train.
6. Look at the settings in the first code cell (Part 0):
   - `QUICK = True` runs on small samples, for a quick test run. The numbers are not valid then.
   - `RUN_APPENDIX = False` stops the run before the extra experiments.
   - `RUN_DEMO = True` starts the Gradio demo in Part 12.
   - `SEED = 45` is used everywhere so every run gives the same numbers.
7. Click **Runtime > Run all**. Colab will ask to connect to your Google Drive, say yes.

The notebook saves everything in your Drive:

```
MyDrive/itm454/
├── yelp_dataset.tar     only needed the first time
├── data/                Yelp sample, SemEval files
├── results/             every result table (.csv) and figures/
└── models/              trained models (DistilBERT and the classical ones)
```

The first run takes a long time, mostly because of the Yelp file and DistilBERT. If a model or a result file is already in your Drive, the notebook loads it instead of training again, so a second run is much faster.

**Running on your own laptop:** it also works with Jupyter. Install the libraries with `pip install -r requirements.txt`, put `yelp_dataset.tar` next to the notebook, and run it. The `data/`, `results/` and `models/` folders are made in the same folder. Without a GPU, DistilBERT training will be very slow.

## The demo

Set `RUN_DEMO = True` in Part 0 and run the notebook (or run Part 12 after the models are trained). Gradio prints a public link that you can open in any browser. You paste a review and get:

- the predicted stars (from DistilBERT, with the confidence),
- a table with Positive / Negative / Mixed / Not mentioned for Food Quality, Service and Atmosphere.

### The mixed-sentence fix (new in v6)

When we tested the demo in v5, we found a problem. For *"The food was great but the service was painfully slow."* it said **Food = Negative**. The demo used the DistilBERT sentiment model, which reads the whole sentence, so the negative half pulled the Food answer down too.

In v6 we changed **only the demo**:

1. Split each sentence at contrast words: but, however, although, though, yet, while, whereas.
2. Find the aspects in each part and score each part on its own.
3. Use the TF-IDF + Logistic Regression sentiment model for this, because it handled mixed sentences better (0.684 vs 0.474, see Task B results below).
4. Combine per aspect like before.

On 6 mixed sentences we wrote ourselves, the old demo got 7 of 13 aspect results right and the new one gets 13 of 13. This is a small test, so it only shows that the fix works on this kind of sentence. It also only helps when there is a contrast word: "great food, slow service" with only a comma is not split. The star rating and all the evaluation numbers below did not change.

## Results

All numbers come from our final run. Every Task A model is tested on the same 10,000 test reviews. We split the data **by restaurant**, so reviews of the same restaurant are never in both train and test.

### Task A: star rating (5 classes)

| Model | Accuracy | Macro-F1 | MAE | Within 1 star | 3-class acc | Binary acc |
|---|---|---|---|---|---|---|
| Naive Bayes | 0.622 | 0.571 | 0.454 | 0.944 | 0.839 | 0.920 |
| Logistic Regression | 0.628 | 0.581 | 0.438 | 0.952 | 0.843 | 0.920 |
| Linear SVM | 0.634 | 0.579 | 0.438 | 0.948 | 0.841 | 0.920 |
| LR + class weights | 0.644 | 0.588 | 0.417 | 0.956 | 0.854 | 0.936 |
| **DistilBERT** | **0.660** | **0.619** | **0.382** | **0.967** | **0.862** | 0.926 |
| GloVe neural network* | 0.522 | 0.472 | 0.638 | 0.886 | 0.759 | 0.836 |

\*From our progress update, tested on a random split.

DistilBERT is our best model, about 3 macro-F1 points better than the best classical model. It is good at 1 star and 5 stars and weak at 2 to 4 stars, and 90.3% of its mistakes are off by exactly one star.

When we make the task easier (LR with class weights), the score goes up a lot. So the model finds positive vs negative well; the hard part is telling neighbouring stars apart.

| Version of the task | Accuracy | Macro-F1 |
|---|---|---|
| 5 classes (normal task) | 0.644 | 0.588 |
| 3 classes (negative / neutral / positive) | 0.834 | 0.740 |
| Binary (1-2 vs 4-5 stars) | 0.959 | 0.944 |
| 1 star vs 5 stars | 0.973 | 0.961 |

### Task B: aspect detection (micro-F1)

| Test set | Rules | TF-IDF + LR | DistilBERT |
|---|---|---|---|
| Our 20% hold-out | 0.710 | 0.796 | **0.870** |
| Official SemEval test (800 sentences) | 0.760 | 0.854 | **0.929** |

### Task B: aspect sentiment (macro-F1, last row is accuracy)

| Evaluation | LR (no clause split) | LR (clause split) | DistilBERT |
|---|---|---|---|
| Gold aspects, all | 0.771 | 0.784 | **0.874** |
| End-to-end, all | 0.757 | 0.769 | **0.818** |
| End-to-end, Service | 0.760 | **0.798** | 0.794 |
| End-to-end, Atmosphere | 0.691 | **0.746** | 0.726 |
| Mixed sentences (19 pairs, accuracy) | - | **0.684** | 0.474 |

### Our own hand-labeled checks

**Humans vs the model on 50 reviews.** Four of us guessed the stars without seeing them. The model (LR + class weights) got 0.78; our scores were 0.60, 0.60, 0.70 and 0.84 (average 0.685). So even people find the exact star hard.

**Task B on 100 real Yelp reviews.** Accuracy was 0.61 for Food Quality, 0.60 for Service and 0.68 for Atmosphere, compared with about 0.82 on SemEval. Two of us labeled the first 20 reviews and only agreed 65% to 85% of the time.

More tables (extra experiments, error causes) are in the report and in the notebook.

## Limitations

- **Neighbouring stars are hard to tell apart.** Most mistakes are off by one star, and even our team only got 60% to 84% right.
- **Mixed reviews.** One review often praises the food and complains about the service. DistilBERT's aspect sentiment was only 47% right on mixed sentences. We fixed this in the demo, but the evaluation numbers still use the sentence-level models.
- **Domain shift.** Task B was trained on short, clean SemEval sentences and drops to 0.60 to 0.68 on long, real Yelp reviews.
- **Small Atmosphere data.** Only 64 Atmosphere test pairs, so those numbers can move a lot.
- **128-token limit.** Long reviews are cut for DistilBERT; 8 of 50 checked errors came from this.
- **Sarcasm is rare** in the data, so we cannot say much about it.
- **Computing.** DistilBERT needs a GPU. We used Colab's free T4 and saved the models to Google Drive.

If we had more time, we would try a longer token limit, give DistilBERT only the part of the sentence about the aspect, and train Task B on some labeled Yelp sentences.

## Team

| Member | Student ID |
|---|---|
| Hengvisoth HOR | 2025488 |
| Soksan HAY | 2025055 |
| Chanroth KHUN | 2025504 |
| Jothimuni Vinod Avishka Mendis | 2024441 |

## References

- Yelp Open Dataset. https://business.yelp.com/data/resources/open-dataset/
- SemEval-2014 Task 4 and SemEval-2016 Task 5 restaurant data, from https://github.com/davidsbatista/Aspect-Based-Sentiment-Analysis
- Sanh, V., Debut, L., Chaumond, J., and Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.
- Pennington, J., Socher, R., and Manning, C. (2014). GloVe: Global Vectors for Word Representation.
- Libraries: scikit-learn, NLTK, Hugging Face Transformers, PyTorch, gensim, Gradio.
