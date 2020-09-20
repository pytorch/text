def process_data_select(data_select):
    if isinstance(data_select, str):
        data_select = (data_select,)
    if not isinstance(data_select, tuple):
        raise TypeError('Expected tuple for data_select, got {} instead.'.format(type(data_select)))
    if not set(data_select).issubset(set(('train', 'test', 'valid'))):
        raise TypeError('Given data_select {} is not supported!'.format(data_select))
    return data_select
