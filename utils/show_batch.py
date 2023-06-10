import matplotlib.pyplot as plt

from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid


def show_batch(dl: DataLoader, row_size: int = 10):

    for images, labels in dl:
        assert isinstance(
            images, Tensor), 'Iter element of Dataloader expected to be of type Tensor'
        assert isinstance(
            labels, Tensor), 'Iter element of Dataloader expected to be of type Tensor'

        assert len(
            images.shape) == 4, 'Iter element of Dataloader expected to be 4D shape (batch_size, color_channels, height, width)'

        _, ax = plt.subplots(figsize=(row_size, row_size))
        ax.set_xticks([])
        ax.set_yticks([])

        plt.imshow(make_grid(images, 10).permute(1, 2, 0).to('cpu'))
        plt.title(str(labels.view(10, -1).__repr__()))
        plt.show()

        break
