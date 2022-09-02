from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class RTY_dataset(CocoDataset):
    CLASSES = ('RTY', 'WYY')

# @DATASETS.register_module
# class XGB_dataset(CocoDataset):
#     CLASSES = ('xgbgaihua', 'meaningless')

@DATASETS.register_module
class FJJ_dataset(CocoDataset):
    CLASSES = ('FJJ', 'WYY')

@DATASETS.register_module
class FJJ_FZK_dataset(CocoDataset):
    CLASSES = ('WYY', 'FJJ', 'FZK')

@DATASETS.register_module
class BPY_dataset(CocoDataset):
    CLASSES = ('WYY', 'BPY')

@DATASETS.register_module
class XGB_dataset(CocoDataset):
    CLASSES = ('XGBGH', 'WYY')

@DATASETS.register_module
class QiGuan_dataset(CocoDataset):
    CLASSES = ('F', 'JZ','XY')

@DATASETS.register_module
class LeiGu_dataset(CocoDataset):
    CLASSES = ('LG')

@DATASETS.register_module
class ZuiTi_dataset(CocoDataset):
    CLASSES = ('ZT')

@DATASETS.register_module
class XMZH_JY_dataset(CocoDataset):
    CLASSES = ('WYY', 'XMZH','XQJY')

@DATASETS.register_module
class ZuiTi_QiGuan_dataset(CocoDataset):
    CLASSES = ('Fei', 'XinYin','JiZhu','ZuiTi')
