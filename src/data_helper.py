import torch
from src.abstract import abs_data

class SyntheticData(abs_data.Data):
    def __init__(self, X, y, Z, Beta, B_per_i, Siblings, Z_ids, ideal_betas, *args, **kwargs) -> None:
        super(SyntheticData, self).__init__(X, y, Z, Beta, B_per_i, Siblings, Z_ids, ideal_betas, *args, **kwargs)
    
    def apply_recourse(self, data_ids, betas:torch.Tensor):
        """Applies recourse to the specified data is and returns the recoursed x
        Args:
            data_ids ([type]): [description]
            betas ([type]): [description]
        Returns:
            [type]: [description]
        """
        _, _, z, _ = self.get_instances(data_ids)
        assert z.shape() == betas.shape(), "Why the hell are the shapes inconsistent?"
        return torch.multiply(z, betas)

    @property
    def _BetaShape(self):
        return [2] * self._Beta.shape[1]
    
    @property
    def _Betadim(self):
        return self._Beta.shape[1]

class ShapenetData(abs_data.Data):
    def __init__(self, X, y, Z, Beta, B_per_i, Siblings, Z_ids, ideal_betas, *args, **kwargs) -> None:
        super(ShapenetData, self).__init__(X, y, Z, Beta, B_per_i, Siblings, Z_ids, ideal_betas, *args, **kwargs)
    
    def apply_recourse(self, data_ids, betas:torch.Tensor,dataset = 2):
        """Applies recourse to the specified data is and returns the recoursed x
        Args:
            data_ids ([type]): [description]
            betas ([type]): [description]
        Returns:
            [type]: [description]
        """
        raise ValueError("For Shapenet we dont need to apply recourse because in the test dataset all the recoursed output is available apriori")


    @property
    def _BetaShape(self):
        return [6, 3, 4]
    
    @property
    def _Betadim(self):
        return self._Beta.shape[1]
        
class GoogleTTSData(abs_data.Data):
    def __init__(self, X, y, Z, Beta, B_per_i, Siblings, Z_ids, ideal_betas, *args, **kwargs) -> None:
        super(GoogleTTSData, self).__init__(X, y, Z, Beta, B_per_i, Siblings, Z_ids, ideal_betas, *args, **kwargs)
    
    def apply_recourse(self, data_ids, betas:torch.Tensor,dataset = 2):
        """Applies recourse to the specified data is and returns the recoursed x
        Args:
            data_ids ([type]): [description]
            betas ([type]): [description]
        Returns:
            [type]: [description]
        """
        raise ValueError("For Shapenet we dont need to apply recourse because in the test dataset all the recoursed output is available apriori")


    @property
    def _BetaShape(self):
        return [3, 4, 5]
    
    @property
    def _Betadim(self):
        return self._Beta.shape[1]



class SyntheticDataHelper(abs_data.DataHelper):
    def __init__(self, train, test, val, train_test_data=None) -> None:
        super(SyntheticDataHelper, self).__init__(train, test, val,  train_test_data)

class ShapenetDataHelper(abs_data.DataHelper):
    def __init__(self, train, test, val, train_test_data=None) -> None:
        super(ShapenetDataHelper, self).__init__(train, test, val, train_test_data)


class GoogleTTSDataHelper(abs_data.DataHelper):
    def __init__(self, train, test, val, train_test_data=None) -> None:
        super(GoogleTTSDataHelper, self).__init__(train, test, val, train_test_data)