import logging
from pathlib import Path

from scipy import stats

import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from utrace import UncertaintyQuantifier
from utrace.utils import flatten_batch, get_coverage
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
         data_path: Path=Path("data/MNIST_example/")):
    
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

    BATCH_SIZE = 1024*10

    # Create an instance of the image classifier model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #classifier = ImageClassifierCNN().to(device)
    classifiers = [ImageClassifierCNN().to(device),
                   ImageClassifierLinear().to(device)]

    models = {
        model.__class__.__name__: {
            'pth': Path('.model') / Path(f'{model.__class__.__name__}.pt'),
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
        logger.info("Model %s already trained.", model_name)

        pt_model = classifier['model']
        pt_model.eval()  # Set the model to evaluation mode

        classes = np.arange(10)

        model = Pytorch_wrapper(pt_model, classes=classes)

        cp = UncertaintyQuantifier(model)

        iterations = 200

        iter_coverages = np.empty(iterations, dtype=float)
        iter_Us = np.empty(iterations, dtype=float)
        iter_alphas = np.empty(iterations, dtype=float)
        iter_accuracies = np.empty(iterations, dtype=float)
        iter_empty_sets = np.empty(iterations, dtype=float)

        test_full_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                            transform=transforms.Compose([transforms.ToTensor(),
                                                                          transforms.Normalize((0.5,), (0.5,)),
                                                                          AddGaussianNoise(0., 1.0)]))

        first_indices_history = []
        for iteration in range(iterations):
            logger.debug("\n---------- Iteration %d ----------", iteration)
            
            cp.reset()

            calibrate_dataset, tune_dataset, test_dataset = random_split(test_full_dataset, [0.2, 0.2, 0.6])

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
            for images, labels in calibrate_loader:
                cp.fit(images, labels, batched=True)

            # Find Uncertainty
            alphas_: list[float] = []
            U_: list[float] = []
            for X_tune, y_tune in tune_loader:
                U, alpha = cp.get_uncertainty_opt(X_tune, y_tune, max_iters=30)
                alphas_.append(alpha)
                U_.append(U)
            alphas = np.array(alphas_)
            Us = np.array(U_)
            alpha = np.nanmean(alphas)
            alpha_std = np.nanstd(alphas)
            U = np.nanmean(Us)
            U_std = np.nanstd(Us)
            iter_alphas[iteration] = alpha
            iter_Us[iteration] = U

            # Find coverage:
            batch_coverages, batch_setsizes = [], []
            total_covered = 0
            for X_n, y_n in test_loader:
                y_p, y_s = cp.predict(X_n, alpha)
                # Filter out the ouputs that are not in the classes
                y_n = flatten_batch(y_n).ravel().astype(int)
                
                total_covered += (y_s[np.arange(len(y_n)), y_n]).sum()
                batch_coverages.append(get_coverage(y_n, y_s))
                batch_setsizes.append(y_s.sum(axis=1))

            coverage = np.array(batch_coverages).mean()

            all_setsizes = np.concatenate(batch_setsizes)
            empty_sets = (all_setsizes == 0).sum()
            iter_empty_sets[iteration] = empty_sets
            # logger.info('Empty sets for model %s: %d out of %d',
            #             model_name, empty_sets, Nv)

            iter_coverages[iteration] = total_covered #coverage

            # Accuracy over the test set
            correct_pix = 0
            total_pix = 0
            for X_v, y_v in test_loader:
                y_p = model(X_v)
                y_v = flatten_batch(y_v).ravel().astype(int)
                y_p = flatten_batch(y_p.cpu()).ravel().astype(int)
                total_pix += len(y_p)
                correct_pix += (y_p==y_v).sum()

            accuracy = correct_pix/total_pix
            iter_accuracies[iteration] = accuracy

        
        print(f"Samples shape: {iter_coverages.shape}")

        U_total = np.mean(iter_Us)
        alpha_total = np.mean(iter_alphas)
        coverage_total = np.mean(iter_coverages)
        accuracy_total = np.mean(iter_accuracies)

        np.save(data_path / Path(f'empty_sets_{model_name}.npy'), iter_empty_sets)
        np.save(data_path / Path(f'coverages_{model_name}.npy'), iter_coverages)
        np.save(data_path / Path(f'Us_{model_name}.npy'), iter_Us)
        np.save(data_path / Path(f'alphas_{model_name}.npy'), iter_alphas)
        np.save(data_path / Path(f'accuracies_{model_name}.npy'), iter_accuracies)

        print(f">U = {U_total}< - alpha = {alpha_total} - >Coverage = {coverage_total}< - Accuracy = {accuracy_total}")
        print("Historial de los primeros índices de calibración:", first_indices_history)

        #N=12000
        Nb = Nc
        Nr = Nv

        a_p = iter_Us.mean()
        l = np.floor((Nb + 1) * a_p)
        a_hyp = Nb + 1 - l
        b_hyp = l

        fig, ax = plt.subplots(figsize=(3.5, 2.5), layout='constrained')

        empirical_data = iter_coverages

        stats.probplot(empirical_data, dist=stats.betabinom, sparams=(Nr, a_hyp, b_hyp), plot=ax)
        fig.savefig(img_path / Path(f'Q-Q_plot_{model_name}.pdf'))

        ks_statistic, p_value_ks = stats.kstest(empirical_data, 'betabinom', args=(Nr, a_hyp, b_hyp))

        fig, ax = plt.subplots(figsize=(3.5, 2.5), layout='constrained')
        ax.hist(iter_empty_sets, density=True, bins=10, alpha=0.6, label='Empty sets number')
        fig.savefig(img_path / Path(f'Empty_sets_{model_name}.pdf'))

        print(f"\n--- Resultados del Test de Kolmogorov-Smirnov - {model_name} ---")
        print(f"Estadístico K-S: {ks_statistic:.4e}")
        print(f"P-valor: {p_value_ks:.4e}")

        # --- 3. Interpretación ---
        alpha_test = 0.05  # Nivel de significancia
        if p_value_ks > alpha_test:
            print(f"\nInterpretación (con alpha={alpha_test}):")
            print("El p-valor es mayor que el nivel de significancia.")
            print("=> No hay evidencia suficiente para rechazar la hipótesis nula.")
            print(f"Los datos podrían seguir una distribución Beta({a_hyp}, {b_hyp}).\n")
        else:
            print(f"\nInterpretación (con alpha={alpha_test}):")
            print("El p-valor es menor o igual al nivel de significancia.")
            print("=> Se rechaza la hipótesis nula.")
            print(f"Los datos probablemente NO siguen una distribución Beta({a_hyp}, {b_hyp}).\n")

        # --- 2. Realizar el test de Cramér-von Mises (versión corregida) ---
        # La función devuelve un objeto con los resultados
        cvm_result = stats.cramervonmises(empirical_data, 'betabinom', args=(Nr, a_hyp, b_hyp))

        # Accedemos a los atributos del objeto
        cvm_statistic = cvm_result.statistic
        p_value_cvm = cvm_result.pvalue


        print(f"\n--- Resultados del Test de Cramér-von Mises - {model_name} ---")
        print(f"Estadístico CvM: {cvm_statistic:.4e}")
        print(f"P-valor: {p_value_cvm:.4e}")

        # --- 3. Interpretación ---
        if p_value_cvm > alpha_test:
            print(f"\nInterpretación (con alpha={alpha_test}):")
            print("El p-valor es mayor que el nivel de significancia.")
            print("=> No hay evidencia suficiente para rechazar la hipótesis nula.")
            print(f"Los datos podrían seguir una distribución Beta({a_hyp}, {b_hyp}).\n\n")
        else:
            print(f"\nInterpretación (con alpha={alpha_test}):")
            print("El p-valor es menor o igual al nivel de significancia.")
            print("=> Se rechaza la hipótesis nula.")
            print(f"Los datos probablemente NO siguen una distribución Beta({a_hyp}, {b_hyp}).\n\n")

        # --- Visualización (muy recomendada) ---
        fig, ax = plt.subplots(figsize=(3.5, 2.5), layout='constrained')

        # Histograma de los datos
        ax.hist(empirical_data, density=True, bins=5, alpha=0.6, label='Histograma de datos')

        # PDF teórica de la distribución Beta
        x = np.linspace(0, 1, 4000)
        center = int(empirical_data.mean())
        x_int = np.linspace(center-12000, center+12000, 24000, dtype=int)
        #pdf_beta = stats.beta.pdf(x, a_hyp, b_hyp)
        pdf_beta = stats.betabinom.pmf(x_int, Nr, a_hyp, b_hyp)
        ax.plot(x_int, pdf_beta, 'r-', lw=2, label=f'PDF de Beta({a_hyp}, {b_hyp})')

        ax.set_title(f'Comparación de datos con la PDF de la Hipótesis - {model_name}')
        ax.set_xlabel('Valor')
        ax.set_ylabel('Densidad')
        ax.legend()
        plt.grid(linestyle='--', alpha=0.6)
        fig.savefig(img_path / Path(f'Binom_Beta_{model_name}.pdf'))


if __name__ == '__main__':

    log_path = Path("log/MNIST_convergence_example.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    img_path = Path("img/MNIST_convergence_example/")
    img_path.mkdir(parents=True, exist_ok=True)
    # Tables
    tab_path = Path("tab/MNIST_convergence_example/")
    tab_path.mkdir(parents=True, exist_ok=True)
    # Data
    data_path = Path("data/MNIST_convergence_example/")
    data_path.mkdir(parents=True, exist_ok=True)

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

    main(train_model=False, img_path=img_path, tab_path=tab_path, data_path=data_path)
    plt.show()
