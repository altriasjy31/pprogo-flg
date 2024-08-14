import openhgnn as hg

def hgat(num_layers, 
         hidden_dims,
         num_classes, **kwargs):
    return hg.HGAT(num_layers=num_layers, 
                   hidden_dims=hidden_dims, 
                   num_classes=num_classes, **kwargs)