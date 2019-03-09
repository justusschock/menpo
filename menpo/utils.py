import torch
import numpy as np 

from collections.abc import Iterable

def convert_tensors(func):
    def wrapper(*args, **kwargs):
        new_args = []
        new_kwargs = {}

        device = None

        for arg in args:
            if isinstance(arg, torch.Tensor):
                device = arg.device
                new_args.append(arg.cpu().detach().numpy())

            else:
                new_args.append(arg)


        for key, val in kwargs.items():
            if isinstance(val, torch.Tensor):
                device = val.device

                new_kwargs[key] = val.cpu().detach().numpy()

            else:
                new_kwargs[key] = val

        ret_vals = func(*new_args, **new_kwargs)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(ret_vals, np.ndarra):
            return torch.from_numpy(ret_vals).to(device)

        elif isinstance(ret_vals, Iterable):
            new_ret_vals = []
            for _val in ret_vals:
                if isinstance(_val, np.ndarray):
                    new_ret_vals.append(torch.from_numpy(_val).to(device))

                else:
                    new_ret_vals.append(_val)

            return new_ret_vals

        elif isinstance(ret_vals, dict):
            new_ret_vals = {}

            for k, v in ret_vals.items():
                if isinstance(val, np.ndarray):
                    new_ret_vals[k] = torch.from_numpy(v).to(device)

                else:
                    new_ret_vals[k] = v

            return new_ret_vals

        else:
            return ret_vals

    return wrapper
        


