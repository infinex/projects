import xgboost as xgb
from sklearn.metrics import classification_report
from revolut.log import _logger


def load_xgb(save_path, n_threads=4):
    """
    load an xgboost model
    Args:
        self:
        bst:
        save_path:
        n_threads:

    Returns:

    """
    bst = xgb.Booster({'nthread': n_threads})  # init model
    bst.load_model(str(save_path))  # load data
    return bst


def save_xgb(bst, save_path):
    """
    save an xgboost model
    Args:
        bst: model
        save_path : path

    Returns:
        bst : xgboost model bst

    """
    bst.save_model(str(save_path))
    return bst


def print_classification_report(y_true, y_predict, features, label):
    """
    Print of Classification Report

    Args:
        y_true (np.array): actual
        y_predict (np.array): predicted true
        label (string): a label to indicate training or evaluation
        features (list): feature label

    Returns:

    """
    classification_dict = classification_report(y_true, y_predict, output_dict=True,
                                                target_names=features)
    for k, v in classification_dict.items():
        if isinstance(v, dict):
            metric_string = ''
            for metrics, result in v.items():
                metric_string += f'{metrics} {result:.2f} '
            _logger.info(f'{label} - {k}: {metric_string}')
        else:
            _logger.info(f'{label} - {k}: {v:.2f}')
