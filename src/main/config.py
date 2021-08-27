from pprint import pprint
from pathlib import Path

__all__ = ['BaseConfig', 'TestConfig']

class BaseConfig:
    __basedir__ = 'data/raw/'
    dataset_name = 'IDRiD'
    # train_img_path = (Path(__basedir__) / dataset_name /'DDR-dataset/lesion_segmentation' / 'train/img', \
    #                 Path(__basedir__) / dataset_name /'DDR-dataset/lesion_segmentation' / 'valid/img')
    # train_mask_path = (Path(__basedir__) / dataset_name / 'DDR-dataset/lesion_segmentation' / 'train/labelcol', \
    #                 Path(__basedir__) / dataset_name / 'DDR-dataset/plesion_segmentation' / 'valid/labelcol')

    # train_img_path = Path(__basedir__) / dataset_name /  'Seg-set/Original_Images'
    # train_mask_path = Path(__basedir__) / dataset_name / 'Seg-set/HardExudate_Masks'
    train_img_path = Path(__basedir__) / dataset_name /  '1. Original Images' / 'a. Training Set'
    train_mask_path = Path(__basedir__) / dataset_name / '2. All Segmentation Groundtruths' / 'a. Training Set'

    # train_img_path = Path(__basedir__) / dataset_name / 'train/image'
    # train_mask_path = Path(__basedir__) / dataset_name / 'train/mask'

    lesion_type = 'SE'
    data_mode = 'binary' 
    gray = False
    augmentation = 'advanced' #options: normal, easy, medium, advanced
    use_ben_transform = False #Good for vessel segmentation 
    scale_size = 1024
    data_type = 'all'  #2 type of input format : whole image or tiles

    #Final
    finetune = False  # Traning only decoder
    num_epochs = 100
    batch_size = 2
    val_batch_size = 2
    learning_rate = 1e-3
    learning_rate_decode = 1e-3
    weight_decay = 1e-5
    is_fp16 = True

    #first
    # model_name = "transunet_r50"
    # model_params = {
    #     'img_size': 1024,
    #     'num_classes':1,
    #     'pretrained': True,
    #     'mlp_dims': 256,
    #     'num_heads': 4,
    #     'num_layers': 4
    # }

    # model_name = "SegFormerStar"
    # model_params = {
    #     "backbone": "mit_b1",
    #     "deep_supervision": True,
    #     "clfhead": True,
    #     "pretrained": True
    # }

    # model_name = "SwinformerStar"
    # model_params = {
    #     "backbone":"swin_tiny",
    #     "clfhead": True,
    #     "deep_supervision": True,
    #     "pretrained": True
    # }
    # model_name = "TransUnet_V2"
    # model_params = {
    #     "img_dim": 1024,
    #     "in_channels": 3,
    #     "classes": 1,
    #     "vit_blocks":4, 
    #     "vit_heads":4,
    #     "vit_dim_linear_mhsa_block": 256
    # }
    # model_name = "medt"
    # model_params = {``
    #     "img_size": 256,
    #     "num_classes":1,
    #     "groups": 8
    # }

    model_name = 'unetplusplusstar'
    model_params = {
        "classes": 1,
        "decoder_attention_type": "scse", 
        "decoder_use_batchnorm": True, 
        "base_dim": 32,
        "encoder_depth": 5, 
        "encoder_name": "BoTSER50_Axial_Imagenet_3",
        "deep_supervision": False,
        "drop_block_prob": 0.0,
        "clf_head": False
    }

    # model_name = "deeplabv3plus_deepsup"
    # model_params = {
    #     "encoder_name": "se_resnet50",
    #     "encoder_weights": "imagenet",
    #     "classes": 1
    # }

    # model_name ="resnet152_fpncat256"
    # model_params = {
    #     "num_classes": 1,
    #     "dropout": 0.1,
    #     "pretrained": True,
    #     "deep_supervision": True
    # }

    # model_name = "unetplusplus_deepsup"
    # model_params = {
    #     "encoder_name": "se_resnet50",
    #     "encoder_weights": "imagenet",
    #     "classes": 1,
    #     "decoder_attention_type": "scse",
    #     "deep_supervision" : True
    # }

    # model_name = "resnet50_attunet"
    # model_params = {    
    #     "num_classes":1,
    #     "drop_rate": 0.1,
    #     "pretrained": True,
    #     "deep_supervision": True
    # }

    # model_name = "hubmap_kaggle"
    # model_params = {
    #     'deep_supervision': True,
    #     'clfhead': True,
    #     'clf_threshold': None,
    #     'pretrained': True
    # }


    # model_name = "Unet3Plus_DS"
    # model_params = {"deep_supervision":True}

    # model_name = 'sa_unet'
    # model_params = {'drop_prob': 0.18}

    # model_name = "resunetplusplus"
    # model_params = None

#     model_name = "TransUnet"
#     model_params = {
#         "in_channels": 3, 
#         "img_dim": 1024, 
#         "classes": 1, 
#         "vit_blocks": 8, 
#         "vit_dim_linear_mhsa_block": 1024
#    }

    # model_name = 'swin_tiny_attunet'
    # model_params = {
    #     'num_classes':1, 
    #     'drop_rate':0.0,
    #     'drop_block_rate': 0.2,
    #     'pretrained':True, 
    #     'freeze_bn':False, 
    #     'freeze_backbone':False
    # }

    # model_name = "vgg_doubleunet"
    # model_params = None
    #Choose at first and no need to change
    metric = "dice"
    mode = "max"

    #Second
    # https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou
    # Should we use IOU loss instead of Dice loss in this case ?
    criterion = {"bce": 0.8, 'log_dice':0.2}
    # criterion_clf = 'bce'
    deep_supervision = False
    if deep_supervision:
        criterion_ds = "bce"

    pos_weights = [500]
    optimizer = "adamw"
    scheduler = "reduce"

    resume_path = "models/IDRiD/SE/Jul25_18_44/checkpoints/best_full.pth"#Resume training

    @classmethod
    def get_all_attributes(cls):    
        d = {}
        son = dict(cls.__dict__)
        dad = dict(cls.__base__.__dict__)

        son.update(dad)
        for k, v in son.items():
            if not k.startswith('__') and k != 'get_all_attributes':
                d[k] = v

        return d

class TestConfig(BaseConfig):
    # test_img_path = Path(BaseConfig.__basedir__) / BaseConfig.dataset_name / 'DDR-dataset/lesion_segmentation' /'test/img'
    # test_mask_path = Path(BaseConfig.__basedir__) / BaseConfig.dataset_name / 'DDR-dataset/lesion_segmentation' / 'test/labelcol'
    test_img_path = Path(BaseConfig.__basedir__) / BaseConfig.dataset_name / '1. Original Images' / 'b. Testing Set'
    test_mask_path = Path(BaseConfig.__basedir__) / BaseConfig.dataset_name / '2. All Segmentation Groundtruths' / 'b. Testing Set'
    # test_img_path = Path(BaseConfig.__basedir__) / BaseConfig.dataset_name / 'test/image'
    # test_mask_path = Path(BaseConfig.__basedir__) / BaseConfig.dataset_name / 'test/mask'
    # test_img_path = Path(BaseConfig.__basedir__) / BaseConfig.dataset_name / 

    out_dir = 'outputs'


if __name__ == '__main__':
    d = dict(BaseConfig.__dict__).copy()
    d_1 = dict(BaseConfig.__base__.__dict__).copy()
    pprint(d)