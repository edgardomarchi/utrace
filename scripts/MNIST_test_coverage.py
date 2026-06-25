import logging
from pathlib import Path
from typing import Union

from _common import setup_example_io

from scipy import stats

import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from utrace import UncertaintyQuantifier
from utrace.utils.pytorch.helpers import flatten_batch
from utrace.utils.pytorch.example_models import (
    ImageClassifierCNN,
    ImageClassifierLinear,
    train_and_save,
)
from utrace.utils.pytorch.model_wrapper import Pytorch_wrapper
from utrace.utils.pytorch.transforms import AddGaussianNoise

logger = logging.getLogger(__name__)


def main(train_model: bool=False,
         img_path: Path=Path("img/MNIST_example/"),
         tab_path: Path=Path("tab/MNIST_example/"),
         data_path: Path=Path("data/MNIST_example/"),
         iterations: int=10):
    
    # Graphics settings for publication quality
    plt.rcParams.update({
        'font.size': 6,
        'font.family': 'serif',
        'text.usetex': False,  
        'figure.dpi': 300,
        'axes.linewidth': 0.5,
        'grid.linewidth': 0.4,
        'lines.linewidth': 1.0,
        'lines.markersize': 3,
        'savefig.bbox': 'tight',
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.transparent': True,
    })

    BATCH_SIZE = 12000

    # Create an instance of the image classifier model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifiers = [ImageClassifierCNN().to(device),
                   ImageClassifierLinear().to(device)]

    models: dict[str, dict[str, Union[Path,torch.nn.Module]]] = {
        model.__class__.__name__: {
            'pth': Path(Path('.model') / f'{model.__class__.__name__}.pt'),
            'model': model
        }
        for model in classifiers
    }

    train_dataset = datasets.MNIST(root='./data', train=False, download=True,
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize((0.5,), (0.5,))]))
    train_base_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    logger.info("Train dataset size: %d", len(train_base_loader.dataset))  #type: ignore

    for model_name, classifier in models.items():
        classifier['pth'].parent.mkdir(parents=True, exist_ok=True)

        if train_model:
            logger.info("Training the %s model...", model_name)
            train_and_save(classifier=classifier['model'], train_dataloader=train_base_loader,
                           model_pth=classifier['pth'], device=device, epochs=20)

        # Load the saved model
        with open(classifier['pth'], 'rb') as f:
            classifier['model'].load_state_dict(torch.load(f))
        logger.info("Model %s already trained.\n", model_name)

        pt_model = classifier['model']
        pt_model.eval()  # Set the model to evaluation mode

        classes = np.arange(10)

        model = Pytorch_wrapper(pt_model, classes=classes)

        cp = UncertaintyQuantifier(N=20000, classes=None, max_batch_size=BATCH_SIZE)

        iter_coverages = np.empty(iterations, dtype=float)
        iter_Us = np.empty(iterations, dtype=float)
        iter_alphas = np.empty(iterations, dtype=float)
        iter_accuracies = np.empty(iterations, dtype=float)
        iter_empty_sets = np.empty(iterations, dtype=float)

        test_full_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                            transform=transforms.Compose([transforms.ToTensor(),
                                                                          transforms.Normalize((0.5,), (0.5,)),
                                                                          AddGaussianNoise(0., 0.5)]))

        data_partition = [0.2, 0.2, 0.6]  # Calibration, Tuning, Test
        Nc, Nt, Nv = np.round(len(test_full_dataset) * np.array(data_partition)).astype(int)

        # For debbuging dataset split randomness:
        first_indices_history = []

        for iteration in range(iterations):
            logger.debug("----------------------------------")
            logger.debug("---------- Iteration %d ----------", iteration)
            logger.debug("----------------------------------\n")
            
            cp.reset()

            calibrate_dataset, tune_dataset, test_dataset = random_split(test_full_dataset, data_partition)

            calibrate_loader = DataLoader(calibrate_dataset, batch_size=BATCH_SIZE, shuffle=True)
            tune_loader = DataLoader(tune_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

            Nc = len(calibrate_loader.dataset)  #type: ignore
            Nt = len(tune_loader.dataset)  #type: ignore
            Nv = len(test_loader.dataset)  #type: ignore

            logger.debug("Calibrate dataset size: %d", Nc)  #type: ignore
            logger.debug("Tune dataset size: %d", Nt)  #type: ignore
            logger.debug("Test dataset size: %d\n", Nv)  #type: ignore

            first_index = calibrate_dataset.indices[0]
            first_indices_history.append(first_index)

            # Calibrate the model
            for X_cal, y_cal in calibrate_loader:
                p_cal = model.predict_proba(X_cal)
                y_cal_arr = flatten_batch(y_cal).ravel().numpy().astype(int)
                cp.calibrate_from_proba(p_cal, y_cal_arr, batched=True)

            # Find Uncertainty: materialize the full tune set, ONE call (alpha/U are
            # non-linear in the data, so per-batch averaging is invalid)
            tune_probs_list, tune_y_list = [], []
            for X_tune, y_tune in tune_loader:
                tune_probs_list.append(model.predict_proba(X_tune))
                tune_y_list.append(flatten_batch(y_tune).ravel().numpy().astype(int))
            tune_probs_all = torch.cat(tune_probs_list, dim=0)
            tune_y_all = np.concatenate(tune_y_list, axis=0)
            U, alpha = cp.get_uncertainty_from_proba(tune_probs_all, tune_y_all, max_iters=30)
            iter_alphas[iteration] = alpha
            iter_Us[iteration] = U

            # Find coverage:
            batch_setsizes = []
            total_covered = 0
            cp.alpha = U  # intentional: threshold parameterized by U (Appendix-A design)
            for X_n, y_n in test_loader:
                p_test = model.predict_proba(X_n)
                y_p, y_s = cp.predict_from_proba(p_test)
                # Filter out the ouputs that are not in the classes
                y_n = flatten_batch(y_n).ravel().numpy().astype(int)

                total_covered += (y_s[np.arange(len(y_n)), y_n]).sum()
                batch_setsizes.append(y_s.sum(axis=1))

            all_setsizes = np.concatenate(batch_setsizes)
            empty_sets = (all_setsizes == 0).sum()

            iter_empty_sets[iteration] = empty_sets
            iter_coverages[iteration] = total_covered

            # Accuracy over the test set
            correct_pix = 0
            total_pix = 0
            for X_v, y_v in test_loader:
                y_p = model(X_v)
                y_v = flatten_batch(y_v).ravel().numpy().astype(int)
                y_p = flatten_batch(y_p.cpu()).ravel().numpy().astype(int)
                total_pix += len(y_p)
                correct_pix += (y_p==y_v).sum()

            accuracy = correct_pix/total_pix
            iter_accuracies[iteration] = accuracy

        U_mean = np.mean(iter_Us)
        alpha_mean = np.mean(iter_alphas)
        coverage_mean = np.mean(iter_coverages) / Nv
        accuracy_mean = np.mean(iter_accuracies)

        np.save(data_path / Path(f'empty_sets_{model_name}.npy'), iter_empty_sets)
        np.save(data_path / Path(f'coverages_{model_name}.npy'), iter_coverages)
        np.save(data_path / Path(f'Us_{model_name}.npy'), iter_Us)
        np.save(data_path / Path(f'alphas_{model_name}.npy'), iter_alphas)
        np.save(data_path / Path(f'accuracies_{model_name}.npy'), iter_accuracies)

        logger.info("\nU = %f - alpha = %f - Mean coverage = %f - Mean Accuracy = %f", U_mean, alpha_mean, coverage_mean, accuracy_mean)
        logger.info("U = %f - alpha = %f - 1-Cov = %f - 1-Acc = %f", U_mean, alpha_mean, 1-coverage_mean, 1-accuracy_mean)
        logger.debug("\nFirst calibration indexes history: %s\n", first_indices_history)

        Nb = Nc
        Nr = Nv

        a_p = U_mean
        l = np.floor((Nb + 1) * a_p)
        a_hyp = Nb + 1 - l
        b_hyp = l

        fig, ax = plt.subplots(figsize=(3.5, 2.5), layout='constrained')

        empirical_data = iter_coverages

        stats.probplot(empirical_data, dist=stats.betabinom, sparams=(Nr, a_hyp, b_hyp), plot=ax)  #type: ignore
        fig.savefig(img_path / Path(f'Q-Q_plot_{model_name}.pdf'))


        fig, ax = plt.subplots(figsize=(3.5, 2.5), layout='constrained')
        ax.hist(iter_empty_sets, density=True, bins=10, alpha=0.6, label='Empty sets number')
        fig.savefig(img_path / Path(f'Empty_sets_{model_name}.pdf'))

        logger.info("\n----- MODEL: %s -----\n", model_name)
        
        logger.info("-- Kolmogorov-Smirnov Test --")
        ks_statistic, p_value_ks = stats.kstest(empirical_data, 'betabinom', args=(Nr, a_hyp, b_hyp))
        
        logger.info("T (K-S): %f",ks_statistic)
        logger.info("p-value: %f", p_value_ks)

        # --- Result ---
        alpha_test = 0.05  # Significance level for the test
        logger.info("Alpha level for the test: %f", alpha_test)
        if p_value_ks > alpha_test:
            logger.info("The p-value is greater than the significance level.")
            logger.info("=> There is not enough evidence to reject the null hypothesis.")
            logger.info("=> Data could follow a BetaBinomial(%f, %f) distribution.\n", a_hyp, b_hyp)
        else:
            logger.info("The p-value is less than or equal to the significance level.")
            logger.info("=> The null hypothesis is rejected.")
            logger.info("=> Data probably does NOT follow a BetaBinomial(%f, %f) distribution.\n", a_hyp, b_hyp)

        logger.info("\n-- Cramér-von Mises Test --")
        cvm_result = stats.cramervonmises(empirical_data, 'betabinom', args=(Nr, a_hyp, b_hyp))
        cvm_statistic = cvm_result.statistic
        p_value_cvm = cvm_result.pvalue

        logger.info("T (CvM): %f", cvm_statistic)
        logger.info("p-value: %f", p_value_cvm)

        # --- Result ---
        if p_value_cvm > alpha_test:
            logger.info("The p-value is greater than the significance level.")
            logger.info("=> There is not enough evidence to reject the null hypothesis.")
            logger.info("=> Data could follow a BetaBinomial(%f, %f) distribution.\n", a_hyp, b_hyp)
        else:
            logger.info("The p-value is less than or equal to the significance level.")
            logger.info("=> The null hypothesis is rejected.")
            logger.info("=> Data probably does NOT follow a BetaBinomial(%f, %f) distribution.\n", a_hyp, b_hyp)

        # --- Visualization ---
        fig, ax = plt.subplots(figsize=(3.5, 2.5), layout='constrained')

        # Histogram
        ax.hist(empirical_data, density=True, bins=5, alpha=0.6, label='Empirical coverages')

        # Theoretical PDF
        # x = np.linspace(0, 1, 4000)
        x_int = np.linspace((coverage_mean*Nv)-12000, (coverage_mean*Nv)+12000, 24000, dtype=int)
        # pdf_beta = stats.beta.pdf(x, a_hyp, b_hyp)
        pdf_beta = stats.betabinom.pmf(x_int, Nr, a_hyp, b_hyp)
        ax.plot(x_int, pdf_beta, 'r-', lw=2, label=f'BetaBinomial({a_hyp}, {b_hyp})')

        ax.set_title(f'{model_name}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        plt.grid(linestyle='--', alpha=0.6)
        fig.savefig(img_path / Path(f'BetaBinom_{model_name}.pdf'))


if __name__ == '__main__':

    img_path, data_path, tab_path, log_path = setup_example_io(__file__)

    main(train_model=False, img_path=img_path, tab_path=tab_path, data_path=data_path, iterations=10)
    plt.show()
