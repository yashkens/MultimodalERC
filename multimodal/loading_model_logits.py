import pandas as pd
import numpy as np
import re
import torch


# simple function to convert string numpy array into list
F = lambda x: [float(d) for d in re.findall(r'-?\d+\.\d+', x)]

def normalize_all_modalities(df: pd.DataFrame):
    """Apply conversion to DataFrame columns"""
    global F
    df.text_pred_labels_logits = df.text_pred_labels_logits.apply(F)
    df.audio_pred_labels_logits = df.audio_pred_labels_logits.apply(F)
    df.video_pred_labels_logits = df.video_pred_labels_logits.apply(F)
    return df


def three_modality_concatenation(modalities: list, df: pd.DataFrame):
    """Create a concatenated tensor from model outputs in the given order
    Args:
        modalities: list like ['text', 'video', 'audio']
        df: DataFrame with model outputs
    Returns:
        Tensor with concatenated model outputs, Tensor with labels
    """
    pred_1 = [x for x in df[f'{modalities[0]}_pred_labels_logits']]
    pred_2 = [x for x in df[f'{modalities[1]}_pred_labels_logits']]
    pred_3 = [x for x in df[f'{modalities[2]}_pred_labels_logits']]
    pred = [x + y + z for x, y, z in zip(pred_1, pred_2, pred_3)]
    pred = torch.Tensor(pred)
    y = torch.Tensor(df.true_labels_num.values)
    return pred, y


def prepare_data(path: str):
    """Prepare data tensors from csv files"""
    # read prepared dataframes
    predictions_train = pd.read_csv(
        path + 'mm_logits_predictions_train.csv')
    predictions_dev = pd.read_csv(
        path + 'mm_logits_predictions_dev.csv')
    predictions_test = pd.read_csv(
        path + 'mm_logits_predictions_test.csv')

    # transform string numpy arrays into lists
    predictions_train = normalize_all_modalities(predictions_train)
    predictions_dev = normalize_all_modalities(predictions_dev)
    predictions_test = normalize_all_modalities(predictions_test)

    # create train & dev & test data tensors
    X_train, y_train = three_modality_concatenation(['text', 'audio', 'video'], 
                                                predictions_train)
    X_dev, y_dev = three_modality_concatenation(['text', 'audio', 'video'], 
                                                    predictions_dev)
    X_test, y_test = three_modality_concatenation(['text', 'audio', 'video'], 
                                                    predictions_test)

    return X_train, y_train, X_dev, y_dev, X_test, y_test
