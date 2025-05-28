import torch

def forward_x8(model, x):
    def _transform(v, op):
        if op == "vflip":
            return torch.flip(v, dims=[2])
        if op == "hflip":
            return torch.flip(v, dims=[3])
        if op == "transpose":
            return v.transpose(2, 3)
        return v

    ops = [[], ["vflip"], ["hflip"], ["transpose"],
           ["vflip", "hflip"], ["vflip", "transpose"],
           ["hflip", "transpose"], ["vflip", "hflip", "transpose"]]

    outputs = []
    for op in ops:
        x_aug = x.clone()
        for o in op:
            x_aug = _transform(x_aug, o)
        sr = model(x_aug)
        for o in reversed(op):
            if o == "transpose":
                sr = sr.transpose(2, 3)
            elif o == "vflip":
                sr = torch.flip(sr, dims=[2])
            elif o == "hflip":
                sr = torch.flip(sr, dims=[3])
        outputs.append(sr)

    return torch.stack(outputs, dim=0).mean(dim=0)