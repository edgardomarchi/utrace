import logging
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms.v2 import RandomPerspective, ElasticTransform

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

# Plots style
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

common_style = {
    'linewidth': 0.8,
    'markersize': 3,
    'elinewidth': 0.6,
    'capsize': 2,
    'markeredgewidth': 0.5,
    'ecolor': 'black',
    'markeredgecolor' : 'black'
}


def main(train_model: bool=False, img_path: Path=Path("img/MNIST_example/"),
         tab_path: Path=Path("tab/MNIST_example/"), data_path: Path=Path("data/MNIST_example/"),
         transform=AddGaussianNoise):

    BATCH_SIZE:int = 300
    splits = [0.02, 0.02, 0.96]  # Calibration, Tuning, Testing

    # Create an instance of the image classifier model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = ImageClassifierCNN().to(device)
    classifiers = [ImageClassifierCNN().to(device),
                   ImageClassifierLinear().to(device)]

    models: dict[str, dict[str, Path | torch.nn.Module]] = {
        model.__class__.__name__: {
            'pth': Path('.model') / Path(f'{model.__class__.__name__}.pt'),
            'model': model
        }
        for model in classifiers
    }

    for model_name, classifier in models.items():
        classifier['pth'].parent.mkdir(parents=True, exist_ok=True)

        if train_model:
            train_dataset = datasets.MNIST(root='./data', train=False, download=True,
                                            transform=transforms.Compose([transforms.ToTensor(),
                                                                          transforms.Normalize((0.5,), (0.5,))]))
            train_base_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            logger.info("Train dataset size: %d", len(train_base_loader.dataset))  #type: ignore
            logger.info("Training the %s model...", model_name)
            train_and_save(classifier=classifier['model'], train_dataloader=train_base_loader,
                           model_pth=classifier['pth'], device=device, epochs=20)

    # Load the saved model
        with open(classifier['pth'], 'rb') as f:
            classifier['model'].load_state_dict(torch.load(f,weights_only=True))
        logger.info("Model %s already trained.", model_name)

        classifier = classifier['model']
        classifier.eval()  # Set the model to evaluation mode

        classes = np.arange(10)

        classifier = Pytorch_wrapper(classifier, classes=classes, device=device)
        
        uq = UncertaintyQuantifier(classifier, classes=classes)

        ################################
        #### Uncertainty evaluation ####
        ################################

        CAL_BATCH_SIZE = BATCH_SIZE
        TUNE_BATCH_SIZE = BATCH_SIZE
        TEST_BATCH_SIZE = BATCH_SIZE

        iter_coverages_ = []
        iter_set_sizes_ = []
        iter_accuracies_ = []
        iter_alphas_ = []
        iter_alphas_std_ = []
        iter_uncertainties_ = []
        iter_uncertainties_std_ = []

        match model_name:
            case 'ImageClassifierCNN':
                noises = np.arange(0, 2.5, 0.5)
                # noises = np.arange(0, 2.5, 0.25)    # AWGN
                # noises = np.arange(0, 1.0, 0.1)   # RandomPerspective
                # noises = np.arange(0, 500.0, 50)  # ElasticTransform
            case 'ImageClassifierLinear':
                noises = np.arange(0, 5, 1.0)
                # noises = np.arange(0, 5, 0.5)       # AWGN
                # noises = np.arange(0, 1.0, 0.1)   # RandomPerspective
                # noises = np.arange(0, 500.0, 50)  # ElasticTransform
            case _:
                noises = np.arange(0, 2.5, 0.15)

        # Dataframe with the results:
        # +
        level0_cols = [model_name]
        level1_cols = ['U', r'$\sigma_U$', 'U_E', r'$\sigma_{U_E}$']

        column_index = pd.MultiIndex.from_product([level0_cols, level1_cols],
                                                    names=['Model', 'Uncertainty'])

        data_df = pd.DataFrame(index=noises, columns=column_index)
        data_df.index.name = 'Noises'
# -

        iterations = 2

        for iteration in range(iterations):
            logger.info("\n---------- Iteration %d (of %d)----------\n", iteration+1, iterations)

            coverages: list[np.float64] = []
            set_sizes: list[np.float64] = []
            accuracies: list[np.float64] = []

            alphas_tuned: list[np.float64] = []
            alphas_tuned_std: list[np.float64] = []
            Us_tuned: list[np.float64] = []
            Us_tuned_std: list[np.float64] = []

            for noise in noises:
                logger.info("\n--------------------------------------------------------------------------------"
                            "\nNoise std: %f\n"
                            "--------------------------------------------------------------------------------\n",
                             noise)

                uq.reset()

                #Accuracy
                logger.info("Calculating empirical accuracy with whole dataset and noise: %f.", noise)
                wholeDataset = datasets.MNIST(root='./data', train=True, download=True,
                                              transform=transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Normalize((0.5,), (0.5,)),
                                                                            # Uncomment the desired transform(s):
                                                                            AddGaussianNoise(0., noise),
                                                                            #RandomPerspective(noise, 1),
                                                                            #ElasticTransform(noise),
                                                                            ]))
                whole_data_loader = DataLoader(wholeDataset, batch_size=BATCH_SIZE, shuffle=True)

                correct_pred = 0
                total_samples = 0
                for test_img, true_classes in whole_data_loader:
                    pred_class = classifier(test_img)
                    gt_classes = flatten_batch(true_classes).numpy().astype(int)
                    pr_classes = flatten_batch(pred_class).cpu().numpy().astype(int)

                    total_samples += len(gt_classes)
                    correct_pred += (pr_classes==gt_classes).sum()

                accuracy = np.float64(correct_pred/total_samples)
                accuracies.append(accuracy)

                # Uncertainties
                logger.info("Estimating uncertainties with noise: %f.", noise)
                cal_dataset, tune_dataset, test_dataset = random_split(wholeDataset, splits)

                calDataLoader = DataLoader(cal_dataset, batch_size=CAL_BATCH_SIZE, shuffle=True)
                tuneDataLoader = DataLoader(tune_dataset, batch_size=TUNE_BATCH_SIZE, shuffle=True)
                testDataLoader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)

                logger.debug("Length of Calibration set: %d", len(calDataLoader.dataset))  #type: ignore
                logger.debug("Length of Tuning set: %d", len(tuneDataLoader.dataset))  #type: ignore
                logger.debug("Length of Test set: %d", len(testDataLoader.dataset))  #type: ignore

                for X_cal, y_cal in calDataLoader:
                    uq.calibrate(X_cal, y_cal, batched=True)

                # Get uncertainty
                alphas_: list[np.float64] = []
                U_: list[np.float64] = []
                for X_tune, y_tune in tuneDataLoader:
                    U, alpha = uq.get_uncertainty_opt(X_tune, y_tune)
                    alphas_.append(alpha)
                    U_.append(U)
                alphas = np.array(alphas_)
                Us = np.array(U_)
                alpha = np.nanmean(alphas)
                alpha_std = np.nanstd(alphas)
                U = np.nanmean(Us)
                U_std = np.nanstd(Us)

                # Test
                batch_coverages, batch_setsizes = [], []
                uq.alpha = U
                for X_n, y_n in testDataLoader:
                    y_p, y_s = uq.predict(X_n, force_non_empty_sets=False)
                    logger.debug("y_p: %s", y_p)
                    logger.debug("y_s: %s", y_s)
                    y_n = flatten_batch(y_n).ravel()#.astype(int)

                    batch_coverages.append(get_coverage(y_n.numpy(), y_s))
                    batch_setsizes.append(y_s.sum(axis=1).max())

                coverage = np.array(batch_coverages).mean()
                # Worst case scenario:
                setsize = np.array(batch_setsizes).max()

                coverages.append(coverage)
                set_sizes.append(setsize)
                alphas_tuned.append(alpha)
                alphas_tuned_std.append(alpha_std)
                Us_tuned.append(U)
                Us_tuned_std.append(U_std)
            
            iter_coverages_.append(coverages)
            iter_set_sizes_.append(set_sizes)
            iter_accuracies_.append(accuracies)
            iter_alphas_.append(alphas_tuned)
            iter_alphas_std_.append(alphas_tuned_std)
            iter_uncertainties_.append(Us_tuned)
            iter_uncertainties_std_.append(Us_tuned_std)

        # Save the results:
        model_data_path = data_path / Path(f'{model_name}')
        model_data_path.mkdir(parents=True, exist_ok=True)
        # +
        iter_coverages = np.array(iter_coverages_)
        mean_coverages=  np.mean(iter_coverages, axis=0)
        np.save(model_data_path / Path('mean_coverages.npy'), mean_coverages)
        std_coverages = np.std(iter_coverages, axis=0)
        np.save(model_data_path / Path('std_coverages.npy'), std_coverages)

        iter_set_sizes = np.array(iter_set_sizes_)
        mean_set_sizes = np.mean(iter_set_sizes, axis=0)

        iter_accuracies = np.array(iter_accuracies_)
        mean_accuracies = np.mean(iter_accuracies, axis=0)
        np.save(model_data_path / Path('mean_accuracies.npy'), mean_accuracies)
        std_accuracies = np.std(iter_accuracies, axis=0)
        np.save(model_data_path / Path('std_accuracies.npy'), std_accuracies)

        iter_alphas = np.array(iter_alphas_)
        mean_alphas = np.mean(iter_alphas, axis=0)

        iter_alphas_std = np.array(iter_alphas_std_)
        mean_alphas_std = np.mean(iter_alphas_std, axis=0)

        iter_uncertainties = np.array(iter_uncertainties_)
        mean_uncertainties = np.mean(iter_uncertainties, axis=0)
        np.save(model_data_path / Path('mean_uncertainties.npy'), mean_uncertainties)
        std_uncertainties = np.std(iter_uncertainties, axis=0)
        np.save(model_data_path / Path('std_uncertainties.npy'), std_uncertainties)

        # Save the noises used:
        noises = np.array(noises)
        np.save(model_data_path / Path('noises.npy'), noises)

        iter_uncertainties_std = np.array(iter_uncertainties_std_)
        mean_uncertainties_std = np.mean(iter_uncertainties_std_, axis=0)
        # -

        # Fill the table:
        model_tab_path = tab_path / Path(f'{model_name}')
        model_tab_path.mkdir(parents=True, exist_ok=True)

        for noise in noises:
            data_df.loc[noise, (model_name,'U')] = mean_uncertainties[noise == noises][0]
            data_df.loc[noise, (model_name,r'$\sigma_U$')] = std_uncertainties[noise == noises][0]
            data_df.loc[noise, (model_name,'U_E')] = 1 - mean_accuracies[noise == noises][0]
            data_df.loc[noise, (model_name,r'$\sigma_{U_E}$')] = std_accuracies[noise == noises][0]
        data_df.to_pickle(model_tab_path / Path(f'{model_name}_class_uncertainty.pkl'))
        dfs=data_df.style.format('{:.3f}')
        dfs.to_latex(model_tab_path / Path('class_uncertainty.tex'), hrules=True, label=f'tab:{model_name}_class_uncertainty',
                     caption='Uncertainty and empirical uncertainty for different noise levels.')
        print(data_df)

        # +
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5), layout='constrained')

        offset = 0.001
        x = np.arange(len(noises))
        x_float = x.astype(float)

        xlabels = [f'{noise:.2f}' for noise in noises]


        plt.setp(ax, xticks=x, xticklabels=xlabels,
                 xlabel=r"$\sigma_n$", ylabel='Uncertainty') #r'$\sigma_n$'
        
        model_img_path = img_path / Path(f'{model_name}')
        model_img_path.mkdir(parents=True, exist_ok=True)

        ax.set_title(model_name + f' - {iterations} iterations')

        ax.errorbar(x_float - offset, 1 - mean_accuracies, yerr=std_accuracies, fmt='-o',
                    color='m', label=r'$\bar{U}_E$',
                    markerfacecolor='magenta', zorder=9, **common_style)
        ax.errorbar(x_float, 1 - mean_coverages, yerr=std_coverages, fmt='--o',
                    color='y', label=r'$1-\overline{Cov}$',
                    markerfacecolor='gold', zorder=10, **common_style)

        ax.errorbar(x_float + offset, mean_uncertainties, yerr=std_uncertainties, fmt='-o',
                    color='r', label=r'$\bar{U}$',
                    markerfacecolor='red', zorder=11, **common_style)


        zoom = 1.8
        ax_inset = zoomed_inset_axes(ax, zoom=zoom, loc='lower right')

        # # Volvemos a plotear los mismos datos DENTRO del inset
        ax_inset.errorbar(x_float - offset, 1 - mean_accuracies, yerr=std_accuracies,
                        fmt='-o', color='m', markerfacecolor='magenta',
                        zorder=9, **common_style)
        ax_inset.errorbar(x_float, 1 - mean_coverages, yerr=std_coverages,
                        fmt='--o', color='darkorange', markerfacecolor='gold',
                        zorder=10, **common_style)
        ax_inset.errorbar(x_float + offset, mean_uncertainties, yerr=std_uncertainties,
                        fmt='-o', color='r', markerfacecolor='red',
                        zorder=11, **common_style)
        
        ax_inset.ticklabel_format(style='sci', axis='y', scilimits=(0,0.1))
        ax_inset.set_xticks([])

        x_idx_l, x_idx_h = (3,3)

        scale=4
        
        y_min = min(mean_uncertainties[x_idx_l] - scale*std_uncertainties[x_idx_l],
                    1 - mean_accuracies[x_idx_l] - scale*std_accuracies[x_idx_l],
                    1 - mean_coverages[x_idx_l] - scale*std_coverages[x_idx_l])
        y_max = max(mean_uncertainties[x_idx_h] + scale*std_uncertainties[x_idx_h],
                    1 - mean_accuracies[x_idx_h] + scale*std_accuracies[x_idx_h],
                    1 - mean_coverages[x_idx_h] + scale*std_coverages[x_idx_h])

        x_scale_seg = 0.5
        # Region to magnify
        x1, x2, y1, y2 = x_idx_l-(x_scale_seg/zoom), x_idx_h+(x_scale_seg/zoom), y_min, y_max # Ajusta estas coordenadas a tu región de interés
        ax_inset.set_xlim(x1, x2)
        ax_inset.set_ylim(y1, y2)

        # No labels in the inset
        plt.yticks(visible=False)

        # Conecting lines between the inset and the main plot
        mark_inset(ax, ax_inset, loc1=1, loc2=3, fc="none", ec="0.5", lw=0.4)

        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        fig.savefig(model_img_path / f'MNIST_stem_{iterations}it.pdf')


if __name__ == '__main__':

    transform_str = "AWGN"

    match transform_str:
        case "AWGN":
            transform = AddGaussianNoise
        case "RandomPerspective":
            transform = RandomPerspective
        case "ElasticTransform":
            transform = ElasticTransform
        case _:
            transform = AddGaussianNoise


    log_path = Path(f"log/{transform_str}/MNIST_example.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    img_path = Path(f"img/MNIST_example/{transform_str}/")
    img_path.mkdir(parents=True, exist_ok=True)
    # Tables
    tab_path = Path(f"tab/MNIST_example/{transform_str}/")
    tab_path.mkdir(parents=True, exist_ok=True)
    # Data
    data_path = Path(f"data/MNIST_example/{transform_str}/")
    data_path.mkdir(parents=True, exist_ok=True)

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
    jax_logger = logging.getLogger('jax')
    jax_logger.setLevel(logging.WARNING)  # Set jax logger to WARNING level
    mpl_logger.setLevel(logging.WARNING)  # Set matplotlib logger to WARNING level

    main(train_model=True, img_path=img_path, tab_path=tab_path, data_path=data_path, transform=transform)
    plt.show()
