import logging
from pathlib import Path

from _common import derive_max_batch_size, precompute_proba, setup_example_io

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms.v2 import RandomPerspective, ElasticTransform

from utrace import UncertaintyQuantifier
from utrace.utils import get_coverage
from utrace.utils.pytorch.helpers import flatten_batch
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

    BATCH_SIZE:int = 12000
    MNIST_TRAIN_SIZE = 60000  # MNIST training set size (fixed; only the noise varies per iteration)
    splits = [0.2, 0.2, 0.6]  # Calibration, Tuning, Testing
    IT:int = 20

    # Create an instance of the image classifier model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
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
        classifier['pth'].parent.mkdir(parents=True, exist_ok=True)  #type: ignore

        if train_model:
            train_dataset = datasets.MNIST(root='./data', train=False, download=True,
                                            transform=transforms.Compose([transforms.ToTensor(),
                                                                          transforms.Normalize((0.5,), (0.5,))]))
            train_base_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            logger.info("Train dataset size: %d", len(train_base_loader.dataset))  #type: ignore
            logger.info("Training the %s model...", model_name)
            train_and_save(classifier=classifier['model'], train_dataloader=train_base_loader,  #type: ignore
                           model_pth=classifier['pth'], device=device, epochs=20)  #type: ignore

    # Load the saved model
        with open(classifier['pth'], 'rb') as f:   #type: ignore
            classifier['model'].load_state_dict(torch.load(f,weights_only=True))  #type: ignore
        logger.info("Model %s already trained.", model_name)

        classifier = classifier['model']  #type: ignore
        classifier.eval()  #type: ignore # Set the model to evaluation mode

        classes = np.arange(10)

        classifier = Pytorch_wrapper(classifier, classes=classes, device=device)

        ################################
        #### Uncertainty evaluation ####
        ################################

        max_batch_size = derive_max_batch_size(splits[1], MNIST_TRAIN_SIZE)
        uqs = []
        for C in classifier.classes_:
            uqs.append(UncertaintyQuantifier(N=20000, classes=[C], max_batch_size=max_batch_size))

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
                noises = np.arange(0, 2.5, 0.25)
                # noises = np.arange(0, 2.5, 0.25)  # AWGN
                # noises = np.arange(0, 1.0, 0.1)   # RandomPerspective
                # noises = np.arange(0, 500.0, 50)  # ElasticTransform
            case 'ImageClassifierLinear':
                noises = np.arange(0, 5, 0.5)
                # noises = np.arange(0, 5, 1)       # AWGN
                # noises = np.arange(0, 1.0, 0.1)   # RandomPerspective
                # noises = np.arange(0, 500.0, 50)  # ElasticTransform
            case _:
                noises = np.arange(0, 2.5, 0.15)

        # Dataframe with the results:
        # +
        level0_cols = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']
        level1_cols = ['U', r'$\sigma_U$', 'U_E', r'$\sigma_{U_E}$']

        column_index = pd.MultiIndex.from_product([level0_cols, level1_cols],
                                                   names=['Classes', 'Uncertainty'])

        data_df = pd.DataFrame(index=noises, columns=column_index)
        data_df.index.name = 'Noises'
# -

        iterations = 2

        for iteration in range(iterations):
            logger.info("\n---------- Iteration %d (of %d)----------\n", iteration+1, iterations)

            coverages: list[list[float]] = [[] for _ in classifier.classes_]
            set_sizes: list[list[float]] = [[] for _ in classifier.classes_]
            accuracies: list[list[float]] = [[] for _ in classifier.classes_]

            alphas_tuned: list[list[float]] = [[] for _ in classifier.classes_]
            alphas_tuned_std: list[list[float]] = [[] for _ in classifier.classes_]
            Us_tuned: list[list[float]] = [[] for _ in classifier.classes_]
            Us_tuned_std: list[list[float]] = [[] for _ in classifier.classes_]

            for noise in noises:
                logger.info("\n--------------------------------------------------------------------------------"
                            "\nNoise std: %f\n"
                            "--------------------------------------------------------------------------------\n",
                             noise)

                for C in classifier.classes_:
                    uqs[C].reset()

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
                # num_workers=0: DataLoader workers fork(), which can deadlock once JAX has started its
                # threads (fork + multithreading hazard, not a JAX bug). spawn or lazy JAX init could
                # re-enable workers — deferred; see docs (TODO) / MIGRATION.md.
                whole_data_loader = DataLoader(wholeDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

                correct_pix = np.zeros_like(classifier.classes_)
                total_pix = np.zeros_like(classifier.classes_)
                for test_img, gt_img in whole_data_loader:
                    pred_img = classifier(test_img)
                    gt_pix = flatten_batch(gt_img).numpy().astype(int)
                    pr_pix = flatten_batch(pred_img).cpu().numpy().astype(int)
                    for C in classifier.classes_:
                        total_pix[C] += (gt_pix==C).sum()
                        correct_pix[C]+=np.logical_and(pr_pix==C, gt_pix==C).sum()

                for C in classifier.classes_:
                    accuracy = correct_pix[C]/total_pix[C]
                    accuracies[C].append(accuracy)
                    logger.info("U_E for class %d: %f", C, 1-accuracy)


                # Uncertainties
                logger.info("Estimating uncertainties with noise: %f.", noise)
                cal_dataset, tune_dataset, test_dataset = random_split(wholeDataset, splits)

                calDataLoader = DataLoader(cal_dataset, batch_size=CAL_BATCH_SIZE, shuffle=True, num_workers=0)
                tuneDataLoader = DataLoader(tune_dataset, batch_size=TUNE_BATCH_SIZE, shuffle=True, num_workers=0)
                testDataLoader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=0)

                logger.debug("Length of Calibration set: %d", len(calDataLoader.dataset))  #type: ignore
                logger.debug("Length of Tuning set: %d", len(tuneDataLoader.dataset))  #type: ignore
                logger.debug("Length of Test set: %d", len(testDataLoader.dataset))  #type: ignore

                # Calibration: batch-outer / class-inner (one model forward per batch)
                for X_cal, y_cal in calDataLoader:
                    p_cal = classifier.predict_proba(X_cal)
                    y_cal_arr = flatten_batch(y_cal).ravel().numpy().astype(int)
                    for C in classifier.classes_:
                        uqs[C].calibrate_from_proba(p_cal, y_cal_arr, batched=True)

                # Precompute tune set (one model forward per batch, shared across classes)
                tune_probs_all, tune_y_all = precompute_proba(tuneDataLoader, classifier)
                tune_y_all = flatten_batch(tune_y_all).ravel().numpy().astype(int)  # COMPAT: remove in the labels .numpy() cleanup step

                # Precompute test set (one model forward per batch, shared across classes)
                test_probs_all, test_y_all = precompute_proba(testDataLoader, classifier)
                test_y_all = flatten_batch(test_y_all).ravel().numpy().astype(int)  # COMPAT: remove in the labels .numpy() cleanup step

                for C in classifier.classes_:
                    # Tuning: one call over the full tune set (not batched/averaged)
                    U, alpha = uqs[C].get_uncertainty_from_proba(tune_probs_all, tune_y_all, max_iters=IT)
                    alpha_std = 0.0
                    U_std = 0.0
                    uqs[C].alpha = alpha

                    # Test: coverage as a global proportion
                    logger.info("Testing class %d with alpha=%f (std=%f) and U=%f (std=%f)...", C, alpha, alpha_std, U, U_std)
                    mask_C = (test_y_all == C)
                    y_n_C = test_y_all[mask_C]
                    y_p, y_s = uqs[C].predict_from_proba(test_probs_all)
                    y_s_C = y_s[mask_C]

                    coverage = get_coverage(y_n_C, y_s_C)
                    if coverage >= 0.99:
                        print(f'!!! Coverage of: {coverage}!!!, Noise: {noise}, Class: {C}, Iter: {iteration}')
                    # Worst case scenario:
                    setsize = y_s_C.sum(axis=1).max() if len(y_s_C) > 0 else 0

                    coverages[C].append(coverage)
                    set_sizes[C].append(setsize)
                    alphas_tuned[C].append(alpha)
                    alphas_tuned_std[C].append(alpha_std)
                    Us_tuned[C].append(U)
                    Us_tuned_std[C].append(U_std)

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
            for C in classes:
                data_df.loc[noise, (f'Class {C}','U')] = mean_uncertainties[C][noise == noises][0]
                data_df.loc[noise, (f'Class {C}',r'$\sigma_U$')] = std_uncertainties[C][noise == noises][0]
                data_df.loc[noise, (f'Class {C}','U_E')] = 1 - mean_accuracies[C][noise == noises][0]
                data_df.loc[noise, (f'Class {C}',r'$\sigma_{U_E}$')] = std_accuracies[C][noise == noises][0]
        data_df.to_pickle(model_tab_path / Path(f'{model_name}_class_uncertainty.pkl'))
        dfs=data_df.style.format('{:.3f}')
        dfs.to_latex(model_tab_path / Path('class_uncertainty.tex'), hrules=True, label=f'tab:{model_name}_class_uncertainty',
                    caption='Class-wise uncertainty and empirical uncertainty for different noise levels.')
        print(data_df)


        if iterations==1:
            U_str = r'${U}$'
            Ue_str = r'${U}_E$'
            Cov_str = r'$1-{Cov}$'
        else:
            U_str = r'$\bar{U}$'
            Ue_str = r'$\bar{U}_E$'
            Cov_str = r'$1-\overline{Cov}$'

        # +
        layout = [
            ['A', 'B', 'C', 'D'],
            ['E', 'F', 'G', 'H'],
            ['.', 'I', 'J', 'L'],
        ]

        fig, axes = plt.subplot_mosaic(layout, figsize=(7, 7/5*3), layout='constrained')
        axs = list(axes.values())

        offset = -0.005
        x = np.arange(len(noises))
        x_float = x.astype(float)

        xlabels = [f'{noise:.2f}' for noise in noises]


        plt.setp(axs, xticks=x, xticklabels=xlabels,
                xlabel=r'$\sigma_n$')
        axs[0].set_ylabel('Uncertainty')
        axs[4].set_ylabel('Uncertainty')
        axs[8].set_ylabel('Uncertainty')

        common_style = {
            'linewidth': 0.8,
            'markersize': 3,
            'elinewidth': 0.6,
            'capsize': 2,
            'markeredgewidth': 0.5,
            'ecolor': 'black',
            'markeredgecolor' : 'black'
        }

        zoom_idx = [(5, 5), (3, 3), (5, 5), (6, 6),
                    (4, 4), (5, 5), (4, 4), (5, 5),
                            (6, 6), (3, 3)]

        match model_name:
            case 'ImageClassifierCNN':
                scales = [5.0, 10.0, 1.5, 1.5,
                          5.0, 1.2, 5.0, 1.5,
                          2.0, 5.0]
        
                zooms = [4, 1.2, 2.0, 2.5,
                         4, 2, 4, 4,
                         2.0, 2.2]
        
                inset_locs = ['lower right', 'lower right', 'center left', 'center left',
                              'lower right', 'center left', 'lower right', 'lower right',
                                             'center left', 'lower right']
   
            case _: #Linear
                scales = [2.0, 4.0, 2.5, 2.0,
                          1.5, 2.2, 1.5, 1.5,
                          2.0, 2.0]

                zooms = [1.8, 1.8, 1.8, 1.7,
                        2.0, 2.0, 1.8, 1.8,
                        1.8, 1.8]
                
                inset_locs = ['center left', 'lower right', 'lower right', 'lower right',
                              'lower right', 'lower right', 'center left', 'center left',
                                            'lower right', 'lower right']

        x_scale_seg = 0.5

        for i in range(11):
            ax = axs[i]
            
            if i == 10:
                continue
            
            scale = scales[i]
            zoom = zooms[i]
            loc = inset_locs[i]
            
            ax.set_title(f'Class {i}')
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0.0, 0.1))

            ax.errorbar(x_float - offset, 1 - mean_accuracies[i], yerr=std_accuracies[i],
                        fmt='-o', color='m', label=r'$\bar{U}_E$',
                        markerfacecolor='magenta', zorder=9, **common_style)

            ax.errorbar(x_float, 1 - mean_coverages[i], yerr=std_coverages[i],
                        fmt='--o', color='darkorange', label=r'$1-\overline{Cov}$',
                        markerfacecolor='gold', zorder=10, **common_style)

            ax.errorbar(x_float + offset, mean_uncertainties[i], yerr=std_uncertainties[i],
                        fmt='-o', color='r', label=r'$\bar{U}$',
                        markerfacecolor='red', zorder=11, **common_style)

            ax.grid(True, linestyle='--', alpha=0.6, zorder=0)

            ax.grid(True, linestyle='--', alpha=0.6)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            # if i==0:
            #     ax_inset = zoomed_inset_axes(ax, zoom=zoom, loc='lower left') # left for tensorflow model
            # else:
            ax_inset = zoomed_inset_axes(ax, zoom=zoom, loc=loc)

            # Insets
            ax_inset.errorbar(x_float - offset, 1 - mean_accuracies[i], yerr=std_accuracies[i],
                            fmt='-o', color='m', markerfacecolor='magenta',
                            zorder=9, **common_style)
            ax_inset.errorbar(x_float, 1 - mean_coverages[i], yerr=std_coverages[i],
                            fmt='--o', color='darkorange', markerfacecolor='gold',
                            zorder=10, **common_style)
            ax_inset.errorbar(x_float + offset, mean_uncertainties[i], yerr=std_uncertainties[i],
                            fmt='-o', color='r', markerfacecolor='red',
                            zorder=11, **common_style)
            
            ax_inset.ticklabel_format(style='sci', axis='y', scilimits=(0,0.1))
            ax_inset.set_xticks([])

            x_idx_l, x_idx_h = zoom_idx[i]

            y_min = min(mean_uncertainties[i,x_idx_l] - scale*std_uncertainties[i,x_idx_l],
                        1 - mean_accuracies[i,x_idx_l] - scale*std_accuracies[i,x_idx_l],
                        1 - mean_coverages[i,x_idx_l] - scale*std_coverages[i,x_idx_l])
            y_max = max(mean_uncertainties[i,x_idx_h] + scale*std_uncertainties[i,x_idx_h],
                        1 - mean_accuracies[i,x_idx_h] + scale*std_accuracies[i,x_idx_h],
                        1 - mean_coverages[i,x_idx_h] + scale*std_coverages[i,x_idx_h])

            x1, x2, y1, y2 = x_idx_l-(x_scale_seg/zoom), x_idx_h+(x_scale_seg/zoom), y_min, y_max
            ax_inset.set_xlim(x1, x2)
            ax_inset.set_ylim(y1, y2)

            plt.xticks(visible=False)
            plt.yticks(visible=False)

            mark_inset(ax, ax_inset, loc1=1, loc2=3, fc="none", ec="0.5", lw=0.4)


        handles, labels = axs[0].get_legend_handles_labels()

        axs[10].axis('off')


        axs[10].legend(handles, labels,
                        loc='center',
                        fontsize='large',
                        frameon=False)
        
        output_img_path = Path(img_path / model_name)
        output_img_path.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_img_path / f'MNIST_c_example_it{iterations}.pdf')

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

    img_path, data_path, tab_path, log_path = setup_example_io(__file__, transform=transform_str)

    main(train_model=False, img_path=img_path, tab_path=tab_path, data_path=data_path, transform=transform)
    plt.show()
