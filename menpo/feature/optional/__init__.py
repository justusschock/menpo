from menpo.base import MenpoMissingDependencyError
from menpo.utils import convert_tensors


try:
    from .vlfeat import dsift as _dsift, fast_dsift as _fast_dsift, \
        vector_128_dsift as _vector_128_dsift, \
        hellinger_vector_128_dsift as _hellinger_vector_128_dsift

    @convert_tensors
    def dsift(*args, **kwargs):
        return _dsift(*args, **kwargs)

    @convert_tensors
    def fast_dsift(*args, **kwargs):
        return _fast_dsift(*args, **kwargs)

    @convert_tensors
    def vector_128_dsift(*args, **kwargs):
        return _vector_128_dsift(*args, **kwargs)

    @convert_tensors
    def hellinger_vector_128_dsift(*args, **kwargs):
        return _hellinger_vector_128_dsift(*args, **kwargs)
    
except MenpoMissingDependencyError:
    pass

# Remove from scope
del MenpoMissingDependencyError
