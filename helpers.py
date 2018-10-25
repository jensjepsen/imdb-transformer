import numpy as np
from dataloader import get_imdb, num2words

def plot_weights(model,windows,b,vocab,vis):
    try:
        weights = model.transformer.blocks[0].attention.weights.to("cpu").numpy()
    except AttributeError:
        print "No weights yet"
        return None
    idx = 1
    text,dims = b.text[0], b.text[1]
    if windows is None:
        windows = [None] * weights.shape[0]
    new_windows = []
    weights = weights[idx]
    dims = dims[idx].item()
    names = num2words(vocab,text[idx].numpy()[:dims])
    weights = weights[:,:dims,:dims]

    for weight, window in zip(weights,windows):
        new_windows.append(vis.heatmap(weight,opts=dict(columnnames=names,rownames=names),win=window))
    return new_windows