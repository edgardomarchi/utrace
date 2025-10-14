#!env -S uv run --python 3.13 --with "utrace[cuda124] @ file://.." --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "bayesian-torch",
#     "matplotlib",
# ]
#
# [[tool.uv.index]]
# url = "https://download.pytorch.org/whl/cu124"
# ///
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
from bayesian_torch.utils.util import predictive_entropy, mutual_information

from utrace.utils.pytorch.example_models import ImageClassifierCNN, train_and_save

logger = logging.getLogger(__name__)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor:torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn(tensor.shape) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean:.2f}, std={self.std:.2f})'


def main(train_model=True, img_path:Path=Path('./img/'), data_path:Path=Path('./data/'), model_path:Path=Path('./saved_models/')) -> None:

    BATCH_SIZE = 1200  # Number of images per batch
    num_classes = 10
    num_monte_carlo = 1000
    splits = [0.2, 0.2, 0.6]
    
    iterations = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageClassifierCNN().to(device)

    model_name = model.__class__.__name__

    if train_model:
        train_dataset = datasets.MNIST(root='./data', train=False, download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize((0.5,), (0.5,))]))
        train_base_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        logger.info("Train dataset size: %d", len(train_base_loader.dataset))  #type: ignore
        logger.info("Training the %s model...", model_name)
        train_and_save(classifier=model, train_dataloader=train_base_loader,
                       model_pth=model_path / "ImageClassifierCNN.pt", device=device, epochs=20)
    
    # Load the saved model
    with open(model_path / "ImageClassifierCNN.pt", 'rb') as f:
        model.load_state_dict(torch.load(f))
        logger.info("Model %s already trained.", model_name)

    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
    }

    dnn_to_bnn(model, const_bnn_prior_parameters)

    noises = np.arange(0, 2.5, 0.5)

    iter_accuracies_ = []
    iter_puncertainties_ = []
    iter_muncertainties_ = []


    for iteration in range(iterations):
        logger.info("---------- Iteration %d (of %d)----------\n", iteration+1, iterations)

        accuracies: list[float] = []
        puncertainties: list[float] = []
        muncertainties: list[float] = []

        for noise in noises:
            wholeDataset = datasets.MNIST(root='./data', train=True, download=True,
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                        transforms.Normalize((0.5,), (0.5,)),
                                                                        AddGaussianNoise(0., noise)]))

            cal_dataset, tune_dataset, test_dataset = random_split(wholeDataset, splits)

            #calDataLoader = DataLoader(cal_dataset, batch_size=BATCH_SIZE, shuffle=True)
            #tuneDataLoader = DataLoader(tune_dataset, batch_size=BATCH_SIZE, shuffle=True)
            testDataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

            x_test, y_test = next(iter(testDataLoader))

            model.eval()
            with torch.no_grad():
                output_mc = []
                for mc_run in range(num_monte_carlo):
                    probs = F.softmax(model(x_test), dim=1)
                    output_mc.append(probs)
                output = torch.stack(output_mc)  
                pred_mean = output.mean(dim=0)
                y_pred = torch.argmax(pred_mean, axis=-1)
                test_acc = (y_pred.data.cpu().numpy() == y_test.data.cpu().numpy()).mean()
            
            predictive_uncertainty = predictive_entropy(output.data.cpu().numpy())
            model_uncertainty = mutual_information(output.data.cpu().numpy())

            pred_classes_mtrx = np.vstack((y_pred.cpu().numpy(),)*num_classes)
            true_classes_mtrx = np.vstack((y_test.cpu().numpy(),)*num_classes)
            coincidences = pred_classes_mtrx==true_classes_mtrx

            class_mtrx = np.repeat(np.arange(num_classes), np.ones(num_classes, dtype=int)*BATCH_SIZE, axis=0).reshape(num_classes, BATCH_SIZE)

            class_sample_mtrx = class_mtrx == true_classes_mtrx
            pc_coincidences = class_sample_mtrx*coincidences

            pc_num_samples = class_sample_mtrx.sum(axis=1)
            logger.debug("Per class num samples: %s\n", pc_num_samples)

            # Use nan to differentiate from class 0
            #class_sample_mtrx[class_sample_mtrx==False]=np.nan
            
            pc_acc = pc_coincidences.sum(axis=1) / pc_num_samples

            pcmu = model_uncertainty * class_sample_mtrx
            pcpu = predictive_uncertainty * class_sample_mtrx

            mean_pcmu = pcmu.sum(axis=1) / pc_num_samples
            mean_pcpu = pcpu.sum(axis=1) / pc_num_samples

            logger.info("---- Noise: %f ----",noise)
            logger.info("Test accuracy: %f", test_acc)
            logger.info("Predictive uncertainty: %s", predictive_uncertainty.mean())
            logger.info("Model uncertainty: %s\n", model_uncertainty.mean())
            logger.info("Per class model uncertainty:\n    %s", mean_pcmu)
            logger.info("Per class predictive uncertianty:\n    %s", mean_pcpu)
            logger.info("Per class empirical uncertainty (1-acc):\n    %s", 1-pc_acc)

            accuracies.append(test_acc)#(pc_acc)
            puncertainties.append(predictive_uncertainty.mean())#(mean_pcpu)
            muncertainties.append(model_uncertainty.mean())#(mean_pcmu)

        iter_accuracies_.append(accuracies)
        iter_puncertainties_.append(puncertainties)
        iter_muncertainties_.append(muncertainties)

    iter_accuracies = np.array(iter_accuracies_)
    iter_puncertainties = np.array(iter_puncertainties_)
    iter_muncertainties = np.array(iter_muncertainties_)

    mean_accuracies = iter_accuracies.mean(axis=0)
    mean_puncertainties = iter_puncertainties.mean(axis=0)
    mean_muncertainties = iter_muncertainties.mean(axis=0)

    std_accuracies = iter_accuracies.std(axis=0)
    std_puncertainties = iter_puncertainties.std(axis=0)
    std_muncertainties = iter_muncertainties.std(axis=0)

    

    ## Plot:
    # +
    fig, ax = plt.subplots(1, 1, figsize=(25, 10))

    offset = 0.001
    x = np.arange(len(noises))
    x_float = x.astype(float)

    xlabels = [f'{noise:.2f}' for noise in noises]


    plt.setp(ax, xticks=x, xticklabels=xlabels,
                xlabel=r'$\sigma_n$', ylabel='Uncertainty')
    
    model_img_path = img_path / Path(f'{model_name}')
    model_img_path.mkdir(parents=True, exist_ok=True)

    ax.set_title(model_name + f' - {iterations} iterations')

    ax.errorbar(x_float - offset, 1 - mean_accuracies, yerr=std_accuracies, fmt='-o',
                color='m', capsize=4, label=r'$\bar{U}_E$',
                ecolor='gray', elinewidth=1,
                markerfacecolor='magenta', markeredgecolor='black', markersize=6, zorder=10)

    # ax.errorbar(x_float, 1 - mean_coverages, yerr=std_coverages, fmt='--o',
    #             color='y', capsize=4, label=r'$1-\overline{Cov}$',
    #             ecolor='gray', elinewidth=1,
    #             markerfacecolor='gold', markeredgecolor='black', markersize=6, zorder=10)

    ax.errorbar(x_float + offset, mean_puncertainties, yerr=std_puncertainties, fmt='-o',
                color='r', capsize=4, label=r'$\bar{U}$',
                ecolor='gray', elinewidth=1,
                markerfacecolor='red', markeredgecolor='black', markersize=6, zorder=10)
    ax.errorbar(x_float + offset, mean_muncertainties, yerr=std_muncertainties, fmt='--o',
                color='g', capsize=4, label=r'$\bar{U}_m$',
                ecolor='gray', elinewidth=1,
                markerfacecolor='green', markeredgecolor='black', markersize=6, zorder=10)

    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout()
    fig.savefig(img_path / f'MNIST_bayesian_approach_{iterations}it.pdf')


if __name__ == "__main__":
    log_path = Path("log/btorch_test.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    img_path = Path("img/MNIST_bayesian_approach/")
    img_path.mkdir(parents=True, exist_ok=True)
    # Tables
    # tab_path = Path("tab/MNIST_class_conditional_example/")
    # tab_path.mkdir(parents=True, exist_ok=True)
    # Data
    data_path = Path("data/MNIST_class_conditional_example/")
    data_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(".saved_models/")
    model_path.mkdir(parents=True, exist_ok=True)

    # Logging configuration
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)  # Global level

    # Console Handler: shows INFO or higher
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s]: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler: only if global level is DEBUG
    if logger.level <= logging.DEBUG:
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("[%(levelname)s - %(filename)s:%(lineno)s - %(funcName)20s()]: %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)  # Set matplotlib logger to WARNING level

    main(train_model=True, img_path=img_path, data_path=data_path, model_path=model_path)
    plt.show()
