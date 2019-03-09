from ._warps_cy import _warp_fast as __warp_fast
from ._daisy import _daisy
from menpo.utils import convert_tensors

@convert_tensors
def _warp_fast(*args, **kwargs):
    return __warp_fast(*args, **kwargs)
    
