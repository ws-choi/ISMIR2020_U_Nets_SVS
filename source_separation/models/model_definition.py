from warnings import warn

from source_separation.models.tfc_tdf_net import TFC_TDF_NET_Framework

def get_class_by_name(model_name):

    if model_name == 'tfc_tdf_net':
        return TFC_TDF_NET_Framework
    else:
        warn('please specify a model name: e.g. --model tfc_tdf_net')
        raise NotImplementedError
