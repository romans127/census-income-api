# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a scikit-learn RandomForestClassifier trained to predict whether a person's annual income exceeds $50,000 using attributes from the 1994 US Census. I kept the forest at 100 trees (n_estimators=100) with random_state=42 so the training run reproduces exactly. The inputs are the standard Adult dataset fields: age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, and native-country. Categorical fields get one-hot encoded, and the salary label is binarized into <=50K and >50K. I built this for WGU D501 Machine Learning DevOps in September 2026, and the model itself is only half the point. The other half is the pipeline around it: a training script, unit tests, a FastAPI inference service, and CI.

## Intended Use

This is a coursework project. The intended use is demonstrating an end-to-end ML deployment workflow: train a model, check slice performance, wrap it in an API, and prove the whole thing works with tests and CI. It is not meant for real decisions about anyone's income, credit, hiring, or benefits. The data is over 30 years old and the model was never tuned, so treat the predictions as demo output and nothing more.

## Training Data

The data is the UCI Census Income dataset, often called the Adult dataset. Barry Becker extracted it from the 1994 US Census database, and it is hosted at https://archive.ics.uci.edu/dataset/20/census+income. Each row describes one person with a mix of demographic and employment attributes, and the label is whether that person earned more than $50,000 in a year. I loaded the CSV, stripped stray whitespace from the column names and string values, and split the data 80/20 with random_state=42. The 80% side was used for training. The positive class (>50K) is the minority, which shows up later in the recall number.

## Evaluation Data

The held-out 20% from that same split is the evaluation set. It goes through the same preprocessing as training, using the encoder and label binarizer fit on the training fold so nothing leaks. Because the split is random with a fixed seed, the evaluation rows come from the same 1994 population as the training rows. There is no separate temporal or geographic holdout.

## Metrics

I used precision, recall, and F1 on the >50K class. On the test set the model scored precision 0.7391, recall 0.6384, and F1 0.6851. The gap between precision and recall tells you the model misses a fair number of actual high earners rather than over-predicting the class.

I also sliced test performance by education level, written to slice_output.txt. A few lines worth calling out: Bachelors sits at F1 0.7449 across 1,053 rows, Masters at F1 0.8381 across 369 rows, while HS-grad drops to F1 0.5114 on 2,085 rows, the largest slice in the set. Some slices look perfect for the wrong reason. Preschool scores F1 1.0000 but only has 10 rows, and 1st-4th scores 1.0000 on 23 rows. With counts that small, a handful of lucky correct calls produces a flawless score, so I would not read anything into those.

## Ethical Considerations

The data describes real people from 1994, and it includes sensitive attributes like race and sex that the model uses directly as features. Income in 1994 reflects the labor market of that era, including its wage gaps across gender and racial groups, so the model can learn and repeat those patterns. The $50K threshold itself is arbitrary and encodes a specific moment in US economic history. If this were anything other than a class exercise, I would want a fairness evaluation across those protected groups and a serious conversation about whether those features belong in the model at all.

## Caveats and Recommendations

Plenty of room to improve here. I did no hyperparameter tuning; the forest runs on its defaults plus the fixed seed. The predicted probabilities are not calibrated, so I would not treat them as confidence scores. There is no fairness mitigation of any kind. The slice output shows HS-grad and Some-college lagging well behind the degree-holding slices, which is worth investigating before trusting the model on any subgroup. If I took this further, I would start with a tuning pass, then calibration, then a proper fairness audit, in that order.
