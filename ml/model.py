import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def save_model(model, path):
    """Serialize a fitted model or preprocessor to disk."""
    joblib.dump(model, path, compress=3)


def load_model(path):
    """Load a fitted model or preprocessor from disk."""
    return joblib.load(path)


def performance_on_categorical_slice(
    df, feature, model, encoder, lb, cat_features, label
):
    """Print precision, recall, and F1 for each value of a categorical feature.

    Inputs
    ------
    df : pd.DataFrame
        Dataframe that still includes the raw categorical column and label.
    feature : str
        Column to slice on.
    model : RandomForestClassifier
        Trained classifier.
    encoder : OneHotEncoder
        Encoder fitted on the training data.
    lb : LabelBinarizer
        Label binarizer fitted on the training data.
    cat_features : list[str]
        Categorical feature names used during training.
    label : str
        Name of the label column.

    Returns
    -------
    lines : list[str]
        One metrics line per non-empty slice.
    """
    lines = []
    for value in sorted(df[feature].dropna().unique(), key=str):
        slice_df = df[df[feature] == value]
        if slice_df.empty:
            continue

        X_slice, y_slice, _, _ = process_data(
            slice_df,
            categorical_features=cat_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        line = (
            f"{feature}={value} | Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | F1: {fbeta:.4f} | Count: {len(slice_df)}"
        )
        print(line)
        lines.append(line)
    return lines
