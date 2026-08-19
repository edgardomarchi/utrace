import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from _common import precompute_softmax, setup_example_io
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from utrace import UncertaintyQuantifier
from utrace.utils.pytorch.example_models import (
    ImageClassifierLinear,
    train_and_save,
)
from utrace.utils.pytorch.helpers import flatten_batch
from utrace.utils.pytorch.model_wrapper import Pytorch_wrapper
from utrace.utils.pytorch.transforms import AddGaussianNoise

logger = logging.getLogger(__name__)


def main(train_model=False, *, img_path: Path, num_sizes=40):

    plt.rcParams.update({
        'font.size': 6,
        'font.family': 'serif',
        'text.usetex': False, # Poner en True si usas LaTeX
        'figure.dpi': 300,
        'axes.linewidth': 0.5, # Grosor de los bordes del gráfico
        'grid.linewidth': 0.4, # Grosor de la grilla
        'lines.linewidth': 1.0, # Grosor de línea principal
        'lines.markersize': 3, # Tamaño de marcador principal
        'savefig.bbox': 'tight',
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.transparent': True,
    })

    BATCH_SIZE = 1024*6

    noise_std = 2.0

    # Create an instance of the image classifier model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = ImageClassifierLinear().to(device)

    model_name = classifier.__class__.__name__
    model_pth = Path('.model') / Path(f'{model_name}.pt')
    model_pth.parent.mkdir(parents=True, exist_ok=True)

    if train_model:
        logger.info("Training the model...")
        train_dataset = datasets.MNIST(root='./data', train=False, download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize((0.5,), (0.5,))]))
        train_base_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        train_and_save(classifier=classifier, train_dataloader=train_base_loader, model_pth=model_pth, epochs=20)

    # Load the saved model
    with open(model_pth, 'rb') as f:
        classifier.load_state_dict(torch.load(f))
    logger.info("Model already trained.")

    C = 10
    classes = np.arange(C)

    classifier = Pytorch_wrapper(classifier, classes=classes)

    #Accuracy
    wholeDataset = datasets.MNIST(root='./data', train=True, download=True,
                                  transform=transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize((0.5,), (0.5,)),
                                                                AddGaussianNoise(0., noise_std)]))
    whole_data_loader = DataLoader(wholeDataset, batch_size=BATCH_SIZE, shuffle=True)

    correct_pix = 0
    total_samples = len(wholeDataset)
    for images, labels in whole_data_loader:
        predicted = classifier(images)
        gt = flatten_batch(labels).numpy().astype(int)
        pr = flatten_batch(predicted.cpu()).numpy().astype(int)
        correct_pix += (pr == gt).sum()

    accuracy = correct_pix/total_samples

    logger.debug("Accuracy: %f", accuracy)

    alphas = np.empty([num_sizes,num_sizes])
    calsizes = np.linspace(0.0005, 0.1, num_sizes)
    tunesizes = np.linspace(0.0005, 0.1, num_sizes)

    # Fixed max_batch_size so the jitted search compiles once for the whole grid;
    # +2 absorbs random_split's remainder round-robin (floor(frac*N)+1 in the worst case).
    max_batch_size = int(np.ceil(tunesizes.max() * total_samples)) + 2
    cp = UncertaintyQuantifier(N=20000, classes=None, max_batch_size=max_batch_size)

    for cs, calsize in enumerate(calsizes):
        for ts, tunesize in enumerate(tunesizes):

            testsize = 1 - calsize - tunesize
            logger.info("Calibrate size: %d - Tune size: %d - Test size %d",
                        int(calsize*total_samples), int(tunesize*total_samples), int(testsize*total_samples))

            # Re-calibrate and re-test with noise:
            calibrate_dataset, tune_dataset, _test_dataset = random_split(wholeDataset, [calsize, tunesize, testsize])

            calibrate_loader = DataLoader(calibrate_dataset, batch_size=BATCH_SIZE, shuffle=True)
            tune_loader = DataLoader(tune_dataset, batch_size=BATCH_SIZE, shuffle=True)

            cp.reset()  # Reset the conformal predictor
            for X_cal, y_cal in calibrate_loader:
                smx_cal = classifier.predict_proba(X_cal)
                y_cal_arr = flatten_batch(y_cal).ravel()
                cp.calibrate(smx_cal, y_cal_arr, batched=True)

            # Tune: materialize the full tune set, ONE call (alphas/U are non-linear in the
            # data, so per-batch averaging is invalid)
            tune_smx_all, tune_y_all = precompute_softmax(tune_loader, classifier)
            tune_y_all = flatten_batch(tune_y_all).ravel()
            U, _alpha = cp.get_uncertainty(tune_smx_all, tune_y_all, max_iters=30)
            alphas[cs, ts] = U  # NOTE: 'alphas' holds U, matching the legacy naming the plots below read

    # Plot the results
    X,Y = np.meshgrid(calsizes*total_samples, tunesizes*total_samples, indexing='ij')
    fig, ax = plt.subplots(1, figsize=(6, 4), subplot_kw={"projection":"3d"})
    accuracies = np.ones_like(alphas) - accuracy
    ax.plot_surface(X, Y, alphas, cmap='viridis', edgecolor='none')  #type: ignore
    ax.plot_surface(X, Y, accuracies, color='m', alpha=0.25, edgecolor='none')  #type: ignore
    ax.set_xlabel("Calibration data size")
    ax.set_ylabel("Tuning data size")
    ax.set_zlabel("U")  #type: ignore
    fig.tight_layout()
    fig.savefig(img_path/'3D_cal_tun_sizes.pdf')


    figures = {
        'tunesizes': {
            'axis':0,
            'sizes':tunesizes,
            'xlabel':'Tuning'
        },
        'calsizes': {
            'axis':1,
            'sizes':calsizes,
            'xlabel':'Calibration'
        },
    }

    for config_dict in figures.values():

        fig, ax = plt.subplots(1, figsize=(3.5, 2.5), layout='constrained')
        axis:int = config_dict['axis'] # type:ignore
        max_alphas = alphas.max(axis=axis)
        min_alphas = alphas.min(axis=axis)
        mean_alphas = alphas.mean(axis=axis)

        sizes:float = config_dict['sizes'] # type:ignore
        ax.plot(sizes*total_samples, mean_alphas, label=r'$\bar{U}$', color='red')
        ax.fill_between(sizes*total_samples, max_alphas, min_alphas, color='red', alpha=0.2)
        ax.plot(sizes*total_samples, np.ones_like(mean_alphas)-accuracy, 'm--', label=r'$\bar{U}_E$')
        ax.set_xlabel(f"{config_dict['xlabel']} data size")
        ax.set_ylabel('U')
        ax.legend()
        fig.savefig(img_path / Path(f"U_vs_{config_dict['xlabel']}_sizes.pdf"))


    for config_dict in figures.values():

        fig, ax = plt.subplots(1, figsize=(3.5, 2.5), layout='constrained')
        axis:int = config_dict['axis'] # type:ignore
        mean_alphas = alphas.mean(axis=axis)
        max_alphas = mean_alphas + alphas.std(axis=axis)
        min_alphas = mean_alphas - alphas.std(axis=axis)

        sizes:float = config_dict['sizes'] # type:ignore
        ax.plot(sizes*total_samples, mean_alphas, label=r'$\bar{U}$', color='red')
        ax.fill_between(sizes*total_samples, max_alphas, min_alphas, color='red', alpha=0.2)
        ax.plot(sizes*total_samples, np.ones_like(mean_alphas)-accuracy, 'm--', label=r'$\bar{U}_E$')
        ax.set_xlabel(f"{config_dict['xlabel']} data size")
        ax.set_ylabel('U')
        ax.legend()
        #fig.tight_layout()
        fig.savefig(img_path / Path(f"U_vs_{config_dict['xlabel']}_sizes_var.pdf"))

if __name__ == '__main__':

    img_path, data_path, tab_path, log_path = setup_example_io(__file__, roots=("img",))

    main(train_model=False, img_path=img_path, num_sizes=8)
    plt.show()
