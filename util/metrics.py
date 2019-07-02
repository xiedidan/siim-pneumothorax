def dice(input, targs, iou=False, eps=1e-8):
    "Dice coefficient metric for binary target. If iou=True, returns iou metric, classic for segmentation problems."
    n = targs.shape[0]

    input = input.view(n,-1)
    targs = targs.view(n,-1)

    intersect = (input * targs).sum().float()
    union = (input + targs).sum().float()

    if not iou:
        return (2. * intersect / union if union > 0 else union.new([1.]).squeeze())
    else:
        return (intersect / (union - intersect + eps) if union > 0 else union.new([1.]).squeeze())
