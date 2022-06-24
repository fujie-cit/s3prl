from .expert import UpstreamExpert as _UpstreamExpert


def opensmile_base(feature_set, *args, **kwargs):
    return _UpstreamExpert(feature_set, *args, **kwargs)

def os_compare_2016(*args, **kwargs):
    return opensmile_base('ComParE_2016', *args, **kwargs)

def os_gemaps_v01b(*args, **kwargs):
    return opensmile_base('GeMAPSv01b', *args, **kwargs)

def os_egemaps_v02(*args, **kwargs):
    return opensmile_base('eGeMAPSv02', *args, **kwargs)

def os_emobase(*args, **kwargs):
    return opensmile_base('emobase', *args, **kwargs)
