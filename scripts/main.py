from args import *
from UNet_monai import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import SpaceNet7DataModule



if __name__ == "__main__":
    args = get_main_args()
    callbacks = []
    model = Unet(args)
    model_ckpt = ModelCheckpoint(dirpath="./", filename="best_model",
                                monitor="dice_mean", mode="max", save_last=True)
    callbacks.append(model_ckpt)
    dm = SpaceNet7DataModule(args)
    trainer = Trainer(callbacks=callbacks, enable_checkpointing=True, max_epochs=args.num_epochs, 
                    enable_progress_bar=True, gpus=1, accelerator="gpu", amp_backend='apex', profiler='simple')

    # train the model
    if args.exec_mode == 'train':
        trainer.fit(model, dm)
    else:
        trainer.predict(model, datamodule=dm, ckpt_path=args.ckpt_path) 
