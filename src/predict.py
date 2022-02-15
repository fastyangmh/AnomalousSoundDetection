# import
from src.project_parameters import ProjectParameters
from src.model import create_model
import torch
from DeepLearningTemplate.data_preparation import parse_transforms, AudioLoader
from DeepLearningTemplate.predict import AudioPredictDataset
from typing import TypeVar, Any

T_co = TypeVar('T_co', covariant=True)
from os.path import isfile
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


# class
class AudioPredictDataset(AudioPredictDataset):
    def __init__(self, root, loader, transform) -> None:
        super().__init__(root, loader, transform)

    def __getitem__(self, index) -> T_co:
        sample = super().__getitem__(index)
        # convert the range of the sample to 0~1
        sample = (sample - sample.min()) / (sample.max() - sample.min())
        return sample


class Predict:
    def __init__(self, project_parameters) -> None:
        self.model = create_model(project_parameters=project_parameters).eval()
        if project_parameters.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.transform = parse_transforms(
            transforms_config=project_parameters.transforms_config)['predict']
        self.device = project_parameters.device
        self.batch_size = project_parameters.batch_size
        self.num_workers = project_parameters.num_workers
        self.classes = project_parameters.classes
        self.loader = AudioLoader(sample_rate=project_parameters.sample_rate)
        self.loss_function = self.model.loss_function
        self.in_chans = project_parameters.in_chans

    def predict(self, inputs) -> Any:
        result = []
        fake_samples = []
        if isfile(path=inputs):
            # predict the file
            sample = self.loader(path=inputs)
            in_chans, _ = sample.shape
            if in_chans != self.in_chans:
                sample = sample.mean(0)
                sample = torch.cat(
                    [sample[None] for idx in range(self.in_chans)])
            # the transformed sample dimension is (1, in_chans, freq, time)
            sample = self.transform(sample)[None]
            # convert the range of the sample to 0~1
            sample = (sample - sample.min()) / (sample.max() - sample.min())
            if self.device == 'cuda' and torch.cuda.is_available():
                sample = sample.cuda()
            with torch.no_grad():
                sample_hat = self.model(sample)
                score = self.loss_function(sample_hat, sample)
                result.append([score.item()])
                fake_samples.append(sample_hat.cpu().data.numpy())
        else:
            # predict the file from folder
            dataset = AudioPredictDataset(root=inputs,
                                          loader=self.loader,
                                          transform=self.transform)
            pin_memory = True if self.device == 'cuda' and torch.cuda.is_available(
            ) else False
            data_loader = DataLoader(dataset=dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers,
                                     pin_memory=pin_memory)
            reduction = self.loss_function.reduction
            self.loss_function.reduction = 'none'
            with torch.no_grad():
                for sample in tqdm(data_loader):
                    if self.device == 'cuda' and torch.cuda.is_available():
                        sample = sample.cuda()
                    sample_hat = self.model(sample)
                    score = self.loss_function(sample_hat, sample)
                    score = score.mean(dim=(1, 2, 3))
                    result.append(score.tolist())
                    fake_samples.append(sample_hat.cpu().data.numpy())
            self.loss_function.reduction = reduction
        result = np.concatenate(result, 0).reshape(-1, 1)
        fake_samples = np.concatenate(fake_samples, 0)
        print(', '.join(self.classes))
        print(result)
        return result, fake_samples


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # predict file
    result = Predict(project_parameters=project_parameters).predict(
        inputs=project_parameters.root)
