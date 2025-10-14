import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from utrace import UncertaintyQuantifier
from utrace.utils import flatten_batch
from utrace.utils.pytorch.example_models import ImageClassifierCNN, ImageClassifierLinear, train_and_save
from utrace.utils.pytorch.model_wrapper import Pytorch_wrapper
from utrace.utils.pytorch.transforms import AddGaussianNoise

logger = logging.getLogger(__name__)


def main(train_model=False, img_path:Path = Path('/img/')):

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

    # train_dataset = datasets.MNIST(root='./data', train=False, download=True,
    #                                transform=transforms.Compose([transforms.ToTensor(),
    #                                                              transforms.Normalize((0.5,), (0.5,)),
    #                                                              AddGaussianNoise(0., 0.5)]))

    # train_base_dataset, _ = random_split(train_dataset, [0.9, 0.1])

    # train_base_data_size = len(train_base_dataset)


    noise_std = 2.0
    # test_full_dataset = datasets.MNIST(root='./data', train=True, download=True,
    #                                    transform=transforms.Compose([transforms.ToTensor(),
    #                                                                  transforms.Normalize((0.5,), (0.5,)),
    #                                                                  AddGaussianNoise(0., noise_std)]))

    # calibrate_dataset, tune_dataset, test_dataset = random_split(test_full_dataset, [0.2, 0.2, 0.6])


    # train_base_loader = DataLoader(train_base_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # calibrate_loader = DataLoader(calibrate_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # tune_loader = DataLoader(tune_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # logger.info("Train dataset size: %d", len(train_base_loader.dataset))  #type: ignore
    # logger.info("Calibrate dataset size: %d", len(calibrate_loader.dataset))  #type: ignore
    # logger.info("Tune dataset size: %d", len(tune_loader.dataset))  #type: ignore
    # logger.info("Test dataset size: %d", len(test_loader.dataset))  #type: ignore


    # Create an instance of the image classifier model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = ImageClassifierCNN().to(device)

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

    cp = UncertaintyQuantifier(classifier)

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
        labels = flatten_batch(labels).astype(int)
        predicted = flatten_batch(predicted.cpu())
        correct_pix += (predicted==labels).sum()

    accuracy = correct_pix/total_samples

    logger.debug("Accuracy: %f", accuracy)

    num_sizes = 40

    alphas = np.empty([num_sizes,num_sizes])
    calsizes = np.linspace(0.0005, 0.1, num_sizes)
    tunesizes = np.linspace(0.0005, 0.1, num_sizes)

    for cs, calsize in enumerate(calsizes):
        for ts, tunesize in enumerate(tunesizes):

            testsize = 1 - calsize - tunesize
            logger.info("Calibrate size: %d - Tune size: %d - Test size %d",
                        int(calsize*total_samples), int(tunesize*total_samples), int(testsize*total_samples))

            # Re-calibrate and re-test with noise:
            calibrate_dataset, tune_dataset, test_dataset = random_split(wholeDataset, [calsize, tunesize, testsize])

            calibrate_loader = DataLoader(calibrate_dataset, batch_size=BATCH_SIZE, shuffle=True)
            tune_loader = DataLoader(tune_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

            cp.reset()  # Reset the conformal predictor
            for images, labels in calibrate_loader:
                cp.fit(images, labels, batched=True)

            # Tune alpha
            alphas_ = []
            for X_tune, y_tune in tune_loader:
                U, alpha = cp.get_uncertainty_opt(X_tune, y_tune, classes, max_iters=30)
                alphas_.append(U)
            alphas[cs,ts] = np.array(alphas_).mean()

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

    for _, config_dict in figures.items():

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
        #fig.tight_layout()
        fig.savefig(img_path / Path(f'U_vs_{config_dict['xlabel']}_sizes.pdf'))


    for _, config_dict in figures.items():

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
        fig.savefig(img_path / Path(f'U_vs_{config_dict['xlabel']}_sizes_var.pdf'))

if __name__ == '__main__':

    log_path = Path("log/data_size_analysis.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    img_path = Path('img/data_size_analysis/')
    img_path.mkdir(parents=True, exist_ok=True)

    # Logging configuration
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)  # Global level

    # Console Handler: shows INFO or higher
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler: only if global level is DEBUG
    if logger.level <= logging.DEBUG:
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)  # Set matplotlib logger to WARNING level

    main(train_model=True, img_path=img_path)
    plt.show()
