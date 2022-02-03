import torch
from torchtext.utils import download_from_url


def count_model_param(nn_model, unit=10**6):
    r"""
    Count the parameters in a model

    Args:
        model: the model (torch.nn.Module)
        unit: the unit of the returned value. Default: 10**6 or M.

    Examples:
        >>> import torch
        >>> import torchtext
        >>> from torchtext.experimental.models.utils import count_model_param
        >>> model = torch.nn.Embedding(100, 200)
        >>> count_model_param(model, unit=10**3)
        >>> 20.
    """
    model_parameters = filter(lambda p: p.requires_grad, nn_model.parameters())
    params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    return params.item() / unit


def load_state_dict_from_url(url, overwrite=False, hash_value=None, hash_type="sha256"):
    try:
        if hash_value:
            return torch.hub.load_state_dict_from_url(url, check_hash=True)
        else:
            return torch.hub.load_state_dict_from_url(url, check_hash=False)
    except ImportError:
        file_path = download_from_url(url, hash_value=hash_value, hash_type=hash_type)
        return torch.load(file_path)


def load_model_from_url(url, overwrite=False, hash_value=None, hash_type="sha256"):
    file_path = download_from_url(url, overwrite=overwrite, hash_value=hash_value, hash_type=hash_type)
    return torch.load(file_path)
