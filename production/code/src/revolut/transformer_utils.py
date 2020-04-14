import dill


def load_transformer(transformer_save_path):
    """
    load a transformer pipeline
    Args:
        transformer_save_path:
    Returns:
        obj : Pipeline
    """
    try:
        with open(transformer_save_path, 'rb') as f:
            obj = dill.load(f)
            return obj
    except Exception as e:
        msg = 'Transformer Error: Unable to load pre fitted transformer from file path %s' % transformer_save_path
        raise ValueError(msg)


def save_transformer(pipeline, save_path):
    """
    save a fitted transformer pipeline

    Args:
        self:
        pipeline: transformer pipeline
        save_path (Path): Path

    Returns:

    """
    save_path.parent.mkdir(exist_ok=True, parents=True)
    with open(save_path, 'wb') as f:
        dill.dump(pipeline, f)
