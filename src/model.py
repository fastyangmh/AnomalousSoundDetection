#import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.model import load_from_checkpoint, BaseModel
import torch.nn as nn
from pl_bolts.models import autoencoders
from os.path import isfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchsummary import summary
import torch


#def
def create_model(project_parameters):
    model = UnsupervisedModel(
        optimizers_config=project_parameters.optimizers_config,
        lr=project_parameters.lr,
        lr_schedulers_config=project_parameters.lr_schedulers_config,
        model_name=project_parameters.model_name,
        in_chans=project_parameters.in_chans,
        input_height=project_parameters.input_height,
        latent_dim=project_parameters.latent_dim,
        loss_function_name=project_parameters.loss_function_name,
        classes=project_parameters.classes)
    if project_parameters.checkpoint_path is not None:
        if isfile(project_parameters.checkpoint_path):
            model = load_from_checkpoint(
                device=project_parameters.device,
                checkpoint_path=project_parameters.checkpoint_path,
                model=model)
        else:
            assert False, 'please check the checkpoint_path argument.\nthe checkpoint_path value is {}.'.format(
                project_parameters.checkpoint_path)
    return model


#class
class UnsupervisedModel(BaseModel):
    def __init__(self, optimizers_config, lr, lr_schedulers_config, model_name,
                 in_chans, input_height, latent_dim, loss_function_name,
                 classes) -> None:
        super().__init__(optimizers_config=optimizers_config,
                         lr=lr,
                         lr_schedulers_config=lr_schedulers_config)
        self.backbone_model = self.create_backbone_model(
            model_name=model_name,
            in_chans=in_chans,
            input_height=input_height,
            latent_dim=latent_dim)
        self.activation_function = nn.Sigmoid()
        self.loss_function = self.create_loss_function(
            loss_function_name=loss_function_name)
        self.classes = classes
        self.stage_index = 0

    def set_in_chans(self, backbone_model, in_chans):
        backbone_model.encoder.conv1 = nn.Conv2d(
            in_channels=in_chans,
            out_channels=backbone_model.encoder.conv1.out_channels,
            kernel_size=backbone_model.encoder.conv1.kernel_size,
            stride=backbone_model.encoder.conv1.stride,
            padding=backbone_model.encoder.conv1.padding,
            bias=backbone_model.encoder.conv1.bias)
        backbone_model.decoder.conv1 = nn.Conv2d(
            in_channels=backbone_model.decoder.conv1.in_channels,
            out_channels=in_chans,
            kernel_size=backbone_model.decoder.conv1.kernel_size,
            stride=backbone_model.decoder.conv1.stride,
            padding=backbone_model.decoder.conv1.padding,
            bias=backbone_model.decoder.conv1.bias)
        return backbone_model

    def create_backbone_model(self, model_name, in_chans, input_height,
                              latent_dim):
        if model_name in dir(autoencoders):
            backbone_model = eval(
                'autoencoders.{}(input_height=input_height, latent_dim=latent_dim)'
                .format(model_name))
            backbone_model = self.set_in_chans(backbone_model=backbone_model,
                                               in_chans=in_chans)
        elif isfile(model_name):
            class_name = self.import_class_from_file(filepath=model_name)
            backbone_model = class_name(in_chans=in_chans,
                                        input_height=input_height,
                                        latent_dim=latent_dim)
        else:
            assert False, 'please check the model_name argument.\nthe model_name value is {}.'.format(
                model_name)
        return backbone_model

    def forward(self, x):
        return self.activation_function(self.backbone_model(x))

    def shared_step(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch=batch)
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch=batch)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        self.log('test_loss', loss)
        reduction = self.loss_function.reduction
        self.loss_function.reduction = 'none'
        loss_step = self.loss_function(x_hat,
                                       x).mean(dim=(1, 2,
                                                    3)).cpu().data.numpy()
        self.loss_function.reduction = reduction
        y_step = y.cpu().data.numpy()
        return {'y': y_step, 'loss': loss_step}

    def test_epoch_end(self, test_outs):
        stages = ['train', 'val', 'test']
        print('\ntest the {} dataset'.format(stages[self.stage_index]))
        print('the {} dataset confusion matrix:'.format(
            stages[self.stage_index]))
        y = np.concatenate([v['y'] for v in test_outs])
        loss = np.concatenate([v['loss'] for v in test_outs])
        figure = plt.figure(figsize=[11.2, 6.3])
        plt.title(stages[self.stage_index])
        for idx, v in enumerate(self.classes):
            score = loss[y == idx]
            sns.kdeplot(score, label=v)
        plt.xlabel(xlabel='Loss')
        plt.legend()
        plt.close()
        self.logger.experiment.add_figure(
            '{} loss density'.format(stages[self.stage_index]), figure,
            self.current_epoch)
        self.stage_index += 1


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create model
    model = create_model(project_parameters=project_parameters)

    # display model information
    summary(model=model,
            input_size=(project_parameters.in_chans,
                        project_parameters.input_height,
                        project_parameters.input_height),
            device='cpu')

    # create input data
    x = torch.ones(project_parameters.batch_size, project_parameters.in_chans,
                   project_parameters.input_height,
                   project_parameters.input_height)

    # get model output
    x_hat = model(x)

    # display the dimension of input and output
    print('the dimension of input: {}'.format(x.shape))
    print('the dimension of output: {}'.format(x_hat.shape))
