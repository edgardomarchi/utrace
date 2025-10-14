import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import beta
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from utrace import UncertaintyQuantifier
from utrace.utils.pytorch.example_models import (
    ImageClassifierCNN,
    ImageClassifierLinear,
    train_and_save,
)
from utrace.utils.pytorch.model_wrapper import Pytorch_wrapper
from utrace.utils.pytorch.transforms import AddGaussianNoise

logger = logging.getLogger(__name__)


def get_beta_dist(mu, sigma, C, num_points=1000):
    mu_n = mu / C
    sigma_n = sigma / C

    nu = mu_n * (1 - mu_n) / sigma_n**2 - 1
    alpha_p = mu_n * nu
    beta_p = (1 - mu_n) * nu

    dist = beta(alpha_p, beta_p)
    x = np.linspace(0, C, num_points)
    x_n = x / C
    pdf_scaled = dist.pdf(x_n) / C

    return x, pdf_scaled


def main(train_model=False, img_path=Path("img/")):

    BATCH_SIZE = 1024*4

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
        train_and_save(classifier=classifier, train_dataloader=train_base_loader,model_pth=model_pth, epochs=20)

    # Load the saved model
    with open(model_pth, 'rb') as f:
        classifier.load_state_dict(torch.load(f))
    logger.info("Model already trained.")

    img_path = img_path / Path(model_name)
    img_path.mkdir(parents=True, exist_ok=True)

    C = 10
    classes = np.arange(C)

    classifier = Pytorch_wrapper(classifier, classes=classes)

    cp = UncertaintyQuantifier(classifier)

    # Tests:
    num_points = 20
    num_noises = 4
    num_lambdas = 4


    bound = np.empty([num_noises, num_points, num_lambdas])
    noises = np.linspace(0, 2.0, num_noises)

    logger.info("Testing set sizes for different alpha values with fixed noises")
    lambdas = np.array([0, 0.1, 1, 2])
    for n, noise_std in enumerate(noises):
        logger.info("Noise std: %.2f", noise_std)

        test_full_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                           transform=transforms.Compose([transforms.ToTensor(),
                                                                         transforms.Normalize((0.5,), (0.5,)),
                                                                         AddGaussianNoise(0., noise_std)]))

        calibrate_dataset, _, test_dataset = random_split(test_full_dataset, [0.2, 0.2, 0.6])

        calibrate_loader = DataLoader(calibrate_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Calibrate the model
        cp.reset()
        for images, labels in calibrate_loader:
            cp.fit(images, labels, batched=True)

        fig, axs = plt.subplots(1, int(num_points/4), figsize=(25, 5))
        axs = axs.flatten()
        fig.suptitle(r'$\sigma_n=$' f'{noise_std:.2f}')

        alphas = np.linspace(0.005, 0.7, num_points)
        for i, alpha in enumerate(alphas):

            setsizes_ = []
            for images, labels in test_loader:
                _, y_s = cp.predict(images, alpha=alpha, force_non_empty_sets=False) # <-- !
                setsizes_.append(y_s.sum(axis=1))

            setsizes = np.concatenate(setsizes_)
            if not i%4:
                axs[i//4].hist(setsizes, density=True, bins=np.linspace(0, C+1, C+2, dtype=int))
                axs[i//4].set_title(r"$\alpha=$"+f'{alpha:.2f}')

            mu = setsizes.mean()
            sigma = setsizes.std()
            for l, lambda_param in enumerate(lambdas):
                bound[n,i,l] = (1-alpha) / (mu+lambda_param*sigma)
                logger.info("Bound: %f & Lambda %d & Alpha: %f & Mean: %f & Variance: %f", bound[n,i,l], lambda_param, alpha, mu, sigma)

            x_b, pdf_scaled = get_beta_dist(mu, sigma, C)
            if not i%4:
                axs[i//4].plot(x_b, pdf_scaled, label='Beta PDF', color='red')
                axs[i//4].set_xlabel('Set size')
        fig.tight_layout()
        fig.savefig(img_path/Path(f'MNIST_setsize_n{noise_std:.2f}.pdf'))

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    for i, bnd in enumerate(bound):
        for l, lambda_param in enumerate(lambdas):
            axs.plot(alphas, bnd[:,l], label=r'$\sigma_n=$'f'{noises[i]:.2f}; 'r'$\lambda=$'f'{lambda_param}')
    axs.set_xlabel(r'$\alpha$')
    axs.set_ylabel(r'$\frac{1-\alpha}{\mu+\sigma}$')
    axs.legend()
    fig.tight_layout()
    fig.savefig(img_path/Path('MNIST_bound_vs_alpha.pdf'))


    logger.info("Testing set sizes for different noises with fixed alpha")

    num_points = 5
    fig, axs = plt.subplots(1, num_points, figsize=(25, 5))
    axs = axs.flatten()

    alpha = 0.3
    for i, noise_std in enumerate(np.linspace(0, 4.0, num_points)):
        logger.info("Noise std: %f", noise_std)

        # Re-calibrate and re-test with noise:
        test_full_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                           transform=transforms.Compose([transforms.ToTensor(),
                                                                         transforms.Normalize((0.5,), (0.5,)),
                                                                         AddGaussianNoise(0., noise_std)]))

        calibrate_dataset, tune_dataset, test_dataset = random_split(test_full_dataset, [0.2, 0.2, 0.6])

        calibrate_loader = DataLoader(calibrate_dataset, batch_size=BATCH_SIZE, shuffle=True)
        tune_loader = DataLoader(tune_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        cp.reset()  # Reset the conformal predictor
        for images, labels in calibrate_loader:
            cp.fit(images, labels, batched=True)

        setsizes_ = []
        for images, labels in test_loader:
            y_p, y_s = cp.predict(images, alpha=alpha)
            setsizes_.append(y_s.sum(axis=1))

        setsizes = np.concatenate(setsizes_)
        axs[i].hist(setsizes, density=True, bins=np.linspace(0, C+1, C+2, dtype=int))
        axs[i].set_title(r"$\sigma_n=$"+f'{noise_std:.2f}')

        mu = setsizes.mean()
        sigma = setsizes.std()
        logger.info("Mean: %f - Variance %f", mu, sigma)

        x_b, pdf_scaled = get_beta_dist(mu, sigma, C)
        axs[i].plot(x_b, pdf_scaled, label=r"$\beta :\; \mu_\beta=$" + f'{mu:.3f}' + r", $\sigma_\beta=$" + f'{sigma:.3f}', color='red')
        axs[i].set_xlabel('Set size')
        axs[i].legend()

    fig.tight_layout()
    fig.savefig(img_path/Path(f'MNIST_setsize_analysis_alpha{alpha:.2f}.pdf'))


    logger.info("Testing set sizes for different noises and alpha that produces average set size of 1")

    fig, axs = plt.subplots(1, num_points, figsize=(25, 5))
    axs = axs.flatten()

    for i, noise_std in enumerate(np.linspace(0, 4.0, num_points)):

        # Re-calibrate and re-test with noise:
        test_full_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Normalize((0.5,), (0.5,)),
                                                            AddGaussianNoise(0., noise_std)]))

        calibrate_dataset, tune_dataset, test_dataset = random_split(test_full_dataset, [0.2, 0.2, 0.6])

        calibrate_loader = DataLoader(calibrate_dataset, batch_size=BATCH_SIZE, shuffle=True)
        tune_loader = DataLoader(tune_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        cp.reset()  # Reset the conformal predictor
        for images, labels in calibrate_loader:
            cp.fit(images, labels, batched=True)


        # Find alpha that produces average set size of 1
        alphas_: list[float] = []
        for images, labels in tune_loader:
            _, alpha = cp.get_uncertainty(images, labels, classes)
            alphas_.append(alpha)

        alphas = np.array(alphas_)
        alpha = np.nanmean(alphas)

        logger.info("Noise std: %f - Alpha found: %f", noise_std, alpha)

        setsizes_ = []
        for images, labels in test_loader:
            y_p, y_s = cp.predict(images, alpha=alpha)
            setsizes_.append(y_s.sum(axis=1))

        setsizes = np.concatenate(setsizes_)
        axs[i].hist(setsizes, density=True, bins=np.linspace(0, (C+1)//2, (C+2)//2, dtype=int))
        axs[i].set_title(r"$\sigma_n=$"+f'{noise_std:.2f} - '+r"$\alpha=$"+f'{alpha:.3f}')

        mu = setsizes.mean()
        sigma = setsizes.std()
        logger.info("Mean: %f - Variance %f", mu, sigma)

        x_b, pdf_scaled = get_beta_dist(mu, sigma, C)
        pdf_n = pdf_scaled / pdf_scaled.max()
        half_length = len(pdf_scaled) // 2
        axs[i].plot(x_b[:half_length], pdf_n[:half_length], label=r"$\beta :\; \mu_\beta=$" + f'{mu:.3f}' + r", $\sigma_\beta=$" + f'{sigma:.3f}', color='red')
        axs[i].set_xlabel('Set size')
        axs[i].legend()

    fig.tight_layout()
    fig.savefig(img_path/Path('MNIST_setsize_1.pdf'))



if __name__ == '__main__':

    log_path = Path("log/setsize_analysis.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    img_path = Path("img/setsize_analysis/")
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
