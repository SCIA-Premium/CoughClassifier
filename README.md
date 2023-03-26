<img align="right" src="https://visitor-badge.laobi.icu/badge?page_id=bictole.coughdetectml&right_color=red">

# CoughClassifier [![Profile][title-img]][profile]

[title-img]:https://img.shields.io/badge/-SCIA--PRIME-red
[profile]:https://github.com/Pypearl

## Authors

[Victor Simonin](https://github.com/Bictole)\
[Alexandre Lemonnier](https://github.com/Alex-Leme)

---

The effective diagnosis of **COVID** could have been an effective tool for limiting coronavirus transmission if our public policies were effective and had implemented the test-trace-and-isolate solution.

Unfortunately, COVID tests require individuals to go to specific locations for testing and take time.

The goal of this project is to develop a **sound classification** model capable of distinguishing a COVID-related cough from a benign cough using recordings of a patient's cough.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To predict if a cough is indicative of a positive or negative COVID case from an **audio file**, simply run the following command:

```python
python predict.py <audio_path>
```

## Observations:

The **classification model** covid_cough_classifier.h5 was generated in the `covid_audio_classification.ipynb` notebook. In this notebook, the librosa library was used to extract the mel-spectrograms from the audio files in the Coswara-Data dataset.

The keras library was then used to build a convolutional layer classification model to predict if a cough is indicative of a COVID-19 case or not. However, the model was not trained on a dataset of sufficient size to predict with high accuracy, and overfitting is present.

Despite this, the model is still able to predict with 70% accuracy if a cough is indicative of a COVID-19 case or not on a test dataset.

The **overfitting** may be due to the lack of significant distinguishing elements in the audio files of the coughs that would allow for COVID-19 prediction in the Coswara-Data dataset, and the size of the dataset used to train the model, which is in the following form:

- Train data : negative (1089) | positive (495)
- Validation data : negative (273) | positive (124)
- Test data : negative (341) | positive (155)

## Sources

- [Coswara-Data](https://github.com/iiscleap/Coswara-Data)
- [Covid-19 Cough classification using machine learning](https://arxiv.org/pdf/2012.01926.pdf)
