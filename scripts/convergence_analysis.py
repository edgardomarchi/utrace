import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from utrace import UncertaintyQuantifier
from utrace.utils import flatten_batch
from utrace.utils.pytorch.example_models import ImageClassifierCNN, train_and_save
from utrace.utils.pytorch.model_wrapper import Pytorch_wrapper
from utrace.utils.pytorch.transforms import AddGaussianNoise

logger = logging.getLogger(__name__)


def main(train_model=False, img_path=Path('img/')):

    BATCH_SIZE = 1024*6
    lambda_ = 0

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
        train_and_save(classifier=classifier, train_dataloader=train_base_loader,model_pth=model_pth)


    # Load the saved model
    with open(model_pth, 'rb') as f:
        classifier.load_state_dict(torch.load(f))
    logger.info("Model already trained.")

    C = 10
    classes = np.arange(C)

    classifier = Pytorch_wrapper(classifier, classes=classes)

    cp = UncertaintyQuantifier(classifier)

    noises = np.array([0, 0.75, 1.25, 2])
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    iters = 10
    num_sizes = 100

    calsizes = np.linspace(0.001, 0.4, num_sizes)
    tunesize = 0.2

    logger.info("Starting the analysis...")

    accuracies = np.empty((iters, len(noises)))
    iter_Us_ = []
    iter_Cov_ = []

    for it in range(iters):
        logger.info("Iteration: %d / %d", it+1, iters)

        Us = np.empty((len(noises), num_sizes))
        Cov = np.empty((len(noises), num_sizes))

        for n, noise_std in enumerate(noises):
            logger.info("Noise std: %f - (%d of %d)", noise_std, n+1, len(noises))

            # Empirical Accuracy
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

            accuracies[it,n] = correct_pix/total_samples

            logger.debug("Empirical uncertaity: Ue=%f", 1-accuracies[it,n])

            for cs, calsize in enumerate(calsizes):
                testsize = 1 - calsize - tunesize
                logger.info("Calibrate size: %d - Tune size: %d - Test size: %d",
                            int(calsize*total_samples), int(tunesize*total_samples), int(testsize*total_samples))
                # Re-calibrate and re-test with noise:
                calibrate_dataset, tune_dataset, test_dataset = random_split(wholeDataset, [calsize, tunesize, testsize])

                calibrate_loader = DataLoader(calibrate_dataset, batch_size=BATCH_SIZE, shuffle=True)
                tune_loader = DataLoader(tune_dataset, batch_size=BATCH_SIZE, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

                cp.reset()  # Reset the conformal predictor
                logger.info("Calibrating the CP...")
                for images, labels in calibrate_loader:
                    cp.fit(images, labels, batched=True)

                # Find uncertainty
                logger.info("Tuning the CP...")
                Us_ = []
                for X_tune, y_tune in tune_loader:
                    U, _ = cp.get_uncertainty(X_tune, y_tune, classes)
                    Us_.append(U)

                Us[n, cs] = np.array(Us_).mean()

                logger.info("Testing U...")
                correct_test_pix = 0
                total_test_samples = len(test_dataset)
                for X_test, y_test in test_loader:
                    predicted = classifier(X_test)
                    labels = flatten_batch(y_test).astype(int)
                    predicted = flatten_batch(predicted.cpu())
                    correct_test_pix += (predicted==labels).sum()
                Cov[n, cs] = 1 - (correct_test_pix/total_test_samples)

        # for each iter:
        iter_Us_.append(Us)
        iter_Cov_.append(Cov)

    iter_Cov=np.array(iter_Cov_)
    mean_Cov = iter_Cov.mean(axis=0)
    min_Cov = iter_Cov.min(axis=0)
    max_Cov = iter_Cov.max(axis=0)

        #for each iter
    mean_acc = np.mean(accuracies,axis=0)
    min_acc = np.min(accuracies,axis=0)
    max_acc = np.max(accuracies,axis=0)

    iter_uncertainties = np.array(iter_Us_)
    min_uncertainties = np.min(iter_uncertainties, axis=0)
    max_uncertainties = np.max(iter_uncertainties, axis=0)
    mean_uncertainties = np.mean(iter_uncertainties, axis=0)

    for n in range(len(noises)):
        # Plot the results
        axs[n].plot(calsizes*total_samples, mean_Cov[n], label=r'$1-\overline{Cov}$', color='green')
        axs[n].fill_between(calsizes*total_samples, max_Cov[n], min_Cov[n], color='green', alpha=0.2)
        axs[n].hlines(y=1-mean_acc[n], xmin=calsizes[0]*total_samples, xmax=calsizes[-1]*total_samples,
                        color='m', linestyle='--', label=r'$\bar{U}_E$', alpha=0.4)
        axs[n].hlines(y=1-max_acc[n], xmin=calsizes[0]*total_samples, xmax=calsizes[-1]*total_samples,
                        color='m', linestyle='--', alpha=0.2)
        axs[n].hlines(y=1-min_acc[n], xmin=calsizes[0]*total_samples, xmax=calsizes[-1]*total_samples,
                        color='m', linestyle='--', alpha=0.2)
        axs[n].plot(calsizes*total_samples, mean_uncertainties[n], label=r'$\bar{U}$', color='red')
        axs[n].fill_between(calsizes*total_samples, min_uncertainties[n], max_uncertainties[n], color='red', alpha=0.2)
        axs[n].set_xlabel("Calibration data size")
        axs[n].set_ylabel('Uncertainty')
        axs[n].set_title(r'$\sigma_n=$'f'{noises[n]}')
        axs[n].legend()

    fig.tight_layout()
    fig.savefig(img_path/Path(f'U_vs_cal_size_it{iters}.pdf'))


if __name__ == '__main__':

    log_path = Path("log/convergence_analysis.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    img_path = Path('img/convergence_analysis/')
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

    main(train_model=False, img_path=img_path)
    plt.show()
