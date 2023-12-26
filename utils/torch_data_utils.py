import torch.utils.data as data_utils
import constants
import torch

class CustomThetaDataset(data_utils.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, data_ids, X, y, transform, *args, **kwargs):
        self.data_ids = data_ids
        self.X = X
        self.y = y
        self.transform = transform


    def __getitem__(self, index):
        data_id, x, y= self.data_ids[index], self.X[index], self.y[index]
        if self.transform is not None:
            x = self.transform(x)
        return data_id, x, y

    def __len__(self):
        return len(self.y)

class CustomXBetaDataset(data_utils.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, data_ids, X, Beta, y, transform, *args, **kwargs):
        self.data_ids = data_ids
        self.X = X
        self.Beta = Beta
        self.y = y
        self.transform = transform


    def __getitem__(self, index):
        data_id, x, beta, y= self.data_ids[index], self.X[index], self.Beta[index], self.y[index]
        if self.transform is not None:
            x = self.transform(x)
        return data_id, x, beta, y

    def __len__(self):
        return len(self.y)

class CustomGrpXBetaDataset(data_utils.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, data_ids_grps, Xgrps, Betagrps, ygrps, transform, *args, **kwargs):
        self.data_ids_grps = data_ids_grps
        self.Xgrps = Xgrps
        self.Betagrps = Betagrps
        self.ygrps = ygrps
        self.transform = transform

    def __getitem__(self, index):
        data_idgrp, xgrp, betagrp, ygrp= self.data_ids_grps[index], self.Xgrps[index], self.Betagrps[index], self.ygrps[index]
        if self.transform is not None:
            xgrp = torch.stack([self.transform(entry) for entry in xgrp])
        return data_idgrp, xgrp, betagrp, ygrp

    def __len__(self):
        return len(self.ygrps)

class CustomGreedyDataset(data_utils.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, data_ids, X, y, transform, *args, **kwargs):
        self.data_ids = data_ids
        self.X = X
        self.y = y
        self.transform = transform


    def __getitem__(self, index):
        data_id, x, y= self.data_ids[index], self.X[index], self.y[index]
        if self.transform is not None:
            x = self.transform(x)
        return data_id, x, y

    def __len__(self):
        return len(self.y)

class CustomPhiDataset(data_utils.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, R_ids, X, Beta, tgt_Beta, transform, *args,  **kwargs):
        self.R_ids = R_ids
        self.X = X
        self.Beta = Beta
        self.tgt_Beta = tgt_Beta
        self.transform = transform

    def __getitem__(self, index):
        r_id, x, beta, tgt_beta = self.R_ids[index], self.X[index], self.Beta[index], self.tgt_Beta[index]

        if self.transform is not None:
            x = self.transform(x)
        return r_id, x, beta, tgt_beta

    def __len__(self):
        return len(self.R_ids)

class CustomPhiGenDataset(data_utils.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, R_ids, X, Beta, Sib_beta, Sij, Sib_losses, transform, *args,  **kwargs):
        self.R_ids = R_ids
        self.X = X
        self.Beta = Beta
        self.Sib_beta = Sib_beta
        self.Sij = Sij
        self.Sib_losses = Sib_losses
        self.transform = transform

    def __getitem__(self, index):
        r_id, x, beta, sib_beta, sij, sib_losses = self.R_ids[index], self.X[index], \
            self.Beta[index], self.Sib_beta[index], self.Sij[index], self.Sib_losses[index]

        if self.transform is not None:
            x = self.transform(x)
        return r_id, x, beta, sib_beta, sij, sib_losses

    def __len__(self):
        return len(self.R_ids)

def get_loader_subset(loader:data_utils.DataLoader, subset_idxs:list, batch_size=None, shuffle=False):
    """Returns a data loader with the mentioned subset indices
    """
    subset_ds = data_utils.Subset(dataset=loader.dataset, indices=subset_idxs)
    if batch_size is None:
        batch_size = loader.batch_size
    return data_utils.DataLoader(subset_ds, batch_size=batch_size, shuffle=shuffle)

def init_loader(ds:data_utils.Dataset, batch_size, shuffle=False, **kwargs):
    if constants.SAMPLER in kwargs:
        return data_utils.DataLoader(ds, batch_size=batch_size, sampler=kwargs[constants.SAMPLER])
    else:
        return data_utils.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def init_grpXBeta_dataset(data_ids, X, y, Beta, B_per_i, transform, **kwargs):
    grp_arr = lambda arr : torch.stack(torch.split(arr, B_per_i))
    xbeta_grpds = CustomGrpXBetaDataset(data_ids_grps=grp_arr(data_ids), Xgrps=grp_arr(X), ygrps=grp_arr(y),
                                     Betagrps=grp_arr(Beta), transform=transform, **kwargs)
    return xbeta_grpds