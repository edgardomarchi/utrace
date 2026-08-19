import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from _common import precompute_softmax, setup_example_io
from scipy.stats import beta
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from utrace import UncertaintyQuantifier
from utrace.utils.pytorch.example_models import (
    ImageClassifierCNN,
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
    MODEL_SEED = 42
    
    torch.manual_seed(MODEL_SEED)
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
        train_generator = torch.Generator().manual_seed(MODEL_SEED)
        train_base_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                       shuffle=True, generator=train_generator)
        train_and_save(classifier=classifier, train_dataloader=train_base_loader, model_pth=model_pth, epochs=10, seed=MODEL_SEED)

    # Load the saved model
    with open(model_pth, 'rb') as f:
        classifier.load_state_dict(torch.load(f))
    logger.info("Model already trained.")

    # Inform the model "signature"
    import hashlib
    state = classifier.state_dict()
    h = hashlib.md5()
    for k in sorted(state.keys()):
        h.update(state[k].detach().cpu().numpy().tobytes())
    logger.info("Model weights hash: %s", h.hexdigest())

    img_path = img_path / Path(model_name)
    img_path.mkdir(parents=True, exist_ok=True)

    C = 10
    classes = np.arange(C)

    classifier = Pytorch_wrapper(classifier, classes=classes)

    # N sized for ~12k calibration samples (20% of 60k MNIST train) with margin
    cp = UncertaintyQuantifier(N=15000)

    # Tests:
    num_points = 20
    num_noises = 4

    noises = np.linspace(0, 2.0, num_noises)

    logger.info("Testing set sizes for different alpha values with fixed noises")

    for n, noise_std in enumerate(noises):
        logger.info("Noise std: %.2f", noise_std)

        test_full_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                           transform=transforms.Compose([transforms.ToTensor(),
                                                                         transforms.Normalize((0.5,), (0.5,)),
                                                                         AddGaussianNoise(0., noise_std)]))

        generator = torch.Generator().manual_seed(24)
        calibrate_dataset, _, test_dataset = random_split(test_full_dataset, [0.2, 0.2, 0.6], generator=generator)

        calibrate_loader = DataLoader(calibrate_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Calibrate: one forward pass over the calibration set, then one call
        cal_softmax, cal_y = precompute_softmax(calibrate_loader, classifier)
        cp.reset()
        cp.calibrate(cal_softmax, cal_y)

        # Precompute test softmax output once per noise level; reused across all alphas
        test_softmax, _ = precompute_softmax(test_loader, classifier)

        fig, axs = plt.subplots(1, int(num_points/4), figsize=(25, 5))
        axs = axs.flatten()
        fig.suptitle(r'$\sigma_n=$' f'{noise_std:.2f}')

        alphas = np.linspace(0.005, 0.7, num_points)
        for i, alpha in enumerate(alphas):

            cp.alpha = alpha
            _, y_s = cp.predict(test_softmax, force_non_empty_sets=False)
            setsizes = y_s.sum(axis=1)

            if not i%4:
                axs[i//4].hist(setsizes, density=True, bins=np.linspace(0, C+1, C+2, dtype=int))
                axs[i//4].set_title(r"$\alpha=$"+f'{alpha:.2f}')

            mu = setsizes.mean()
            sigma = setsizes.std()

            x_b, pdf_scaled = get_beta_dist(mu, sigma, C)
            if not i%4:
                axs[i//4].plot(x_b, pdf_scaled, label='Beta PDF', color='red')
                axs[i//4].set_xlabel('Set size')
        fig.tight_layout()
        fig.savefig(img_path/Path(f'MNIST_setsize_n{noise_std:.2f}.pdf'))


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
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        cal_softmax, cal_y = precompute_softmax(calibrate_loader, classifier)
        cp.reset()
        cp.calibrate(cal_softmax, cal_y)

        cp.alpha = alpha
        test_softmax, _ = precompute_softmax(test_loader, classifier)
        _, y_s = cp.predict(test_softmax)
        setsizes = y_s.sum(axis=1)

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

        cal_softmax, cal_y = precompute_softmax(calibrate_loader, classifier)
        cp.reset()
        cp.calibrate(cal_softmax, cal_y)

        # Find alpha that produces average set size of 1:
        # materialize the full tune set, then ONE binary search — no per-batch averaging
        tune_softmax, tune_y = precompute_softmax(tune_loader, classifier)
        U, alpha = cp.get_uncertainty(tune_softmax, tune_y)
        cp.alpha = alpha

        logger.info("Noise std: %f - Alpha found: %f", noise_std, alpha)

        test_softmax, _ = precompute_softmax(test_loader, classifier)
        _, y_s = cp.predict(test_softmax)
        setsizes = y_s.sum(axis=1)

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

    img_path, data_path, tab_path, log_path = setup_example_io(__file__)

    main(train_model=False, img_path=img_path)
    plt.show()
