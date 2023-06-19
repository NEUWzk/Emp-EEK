EOS_idx = 2
SOS_idx = 3

def clean_preds(preds):
    res = []
    preds = preds.cpu().tolist()
    for pred in preds:
        if EOS_idx in pred:
            ind = pred.index(EOS_idx) + 1  # end_idx included
            pred = pred[:ind]
        if len(pred) == 0:
            continue
        if pred[0] == SOS_idx:
            pred = pred[1:]
        res.append(pred)
    return res