"""Evaluate U-TraCE uncertainty estimation on a real black-box segmentation model.

This script reproduces the cardiac MRI segmentation use case of the U-TraCE paper
(Marchi & Liebl 2026, Mach. Learn.: Sci. Technol. 7 015017) — figures 13-16 and the
class-wise uncertainty tables B1/B2. It applies the per-class U-TraCE procedure to a
pre-trained cardiac segmentation model treated as a true black box: the model is never
retrained or modified, and uncertainty is estimated solely from its softmax outputs over
unseen reference data.

MODEL UNDER TEST (the "black box")
    KM — a residual U-Net for left-ventricle segmentation (Kerfoot et al. 2019),
    pre-trained on UK Biobank cardiac MRI and used here without any retraining. It is
    loaded as a MONAI bundle. The model is third-party and is NOT redistributed with this
    repository; it must be obtained separately.
        Source: obtained via the misas toolbox (Ankenbrand et al. 2021).
        Download: https://huggingface.co/MONAI/ventricular_short_axis_3label/tree/0.3.4
    Expected location (see paths below): a MONAI bundle directory containing
    `train.json` and `models/model.pt`.

REFERENCE DATASET (ground truth)
    ACDC — the Automatic Cardiac Diagnosis Challenge dataset (Bernard et al. 2018). It is
    unseen by the model and is used only as reference/ground truth for calibration,
    tuning, and verification. The dataset is third-party and is NOT redistributed with
    this repository; it must be obtained separately under its own terms.
        Download: https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb

EXPECTED PATHS (adjust to your environment)
    The script assumes both artifacts sit at fixed locations relative to its working
    directory:
        dataset_path  = ../../data/ACDC/
        ai_model_path = ../../models/ventricular_short_axis/   (MONAI bundle root)
    Edit these at the top of `main()` if your layout differs.

EXTRA DEPENDENCIES (beyond the utrace core)
    This script needs packages that are NOT required by the utrace core:
        monai     — to load the model bundle
        nibabel   — to read the NIfTI volumes (via the ACDC dataset wrapper)
        pandas    — to assemble the LaTeX result tables
        torchvision, matplotlib

USAGE NOTES
    - Set USE_JAX appropriately in the environment (see config / .env); the core import
      depends on it (see MIGRATION.md).
    - Per-class CP buffer sizes (N) are computed from the data: a one-time pass counts
      per-class pixels, then N is sized to the calibration fraction with a safety margin.
      This is required for segmentation, where pixel counts reach tens of millions.
    - The degradation transform defaults to AWGN to match the paper's validation. Random
      Perspective and Elastic Transform variants are available (see __main__).

REFERENCES
    Kerfoot, E., Clough, J., Oksuz, I., Lee, J., King, A. P., & Schnabel, J. A. (2019).
        Left-Ventricle Quantification Using Residual U-Net. In Statistical Atlases and
        Computational Models of the Heart (pp. 371-380). Springer. ISBN 978-3-030-12029-0.
    Bernard, O., et al. (2018). Deep Learning Techniques for Automatic MRI Cardiac
        Multi-Structures Segmentation and Diagnosis: Is the Problem Solved?
        IEEE Transactions on Medical Imaging, 37(11), 2514-2525.
        doi:10.1109/TMI.2018.2837502
"""

import logging
from pathlib import Path

from _common import setup_example_io

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap

import math

import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from monai.bundle.utils import load_bundle_config
from torchvision import transforms
from torchvision.transforms.v2 import RandomPerspective, ElasticTransform


from utrace import UncertaintyQuantifier
from utrace.utils import get_coverage
from utrace.utils.pytorch.helpers import flatten_batch
from utrace.utils.pytorch.dataset_wrapper import (
    get_ACDC_cal_tun_tst_dataloaders,
    get_ACDC_dataloader,
)
from utrace.utils.pytorch.model_wrapper import Pytorch_wrapper

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

################# Define Custom Colormaps ###################
color1 = np.array([129 / 256, 154 / 256, 193 / 256, 1.0])
color2 = np.array([193 / 256, 168 / 256, 129 / 256, 1.0])
color3 = np.array([193 / 256, 129 / 256, 154 / 256, 1.0])
color4 = np.array([193 / 256, 129 / 256, 154 / 256, 0])

mapping = np.linspace(0, 3, 256)
newcolors = np.empty((256, 4))
newcolors[mapping > 2] = color3
newcolors[mapping <=  2] = color2
newcolors[mapping <= 1] = color1
newcolors[mapping < 1] = color4

# Make the colormap from the listed colors
my_ai_colormap = ListedColormap(newcolors)

class MinMaxNormalize:
    def __call__(self, x):
        return (x - x.min()) / (x.max() - x.min())


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.shape) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def main(img_path: Path=Path("img/ACDC_example/"),
         tab_path: Path=Path("tab/ACDC_example/"), data_path: Path=Path("data/ACDC_example/"),
         transform=AddGaussianNoise):
    
    # Paths to the externally-obtained model and dataset (NOT shipped with the repo).
    # See the module docstring for sources and expected layout. Adjust to your environment.
    dataset_path = Path('../../data/ACDC/')
    ai_model_path = Path('../../models/ventricular_short_axis')
    model_name = 'KM'

    BATCH_SIZE = 200  # Number of images per batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ai_model_file = ai_model_path / Path("models/model.pt")

    # Load the pre-trained model ("""black-box""")
    parser = load_bundle_config(str(ai_model_path), "train.json")
    net = parser.get_parsed_content("network_def")
    net.load_state_dict(
        torch.load(ai_model_file, map_location=device))

    # Wrap the model:
    classes = np.arange(4)
    model = Pytorch_wrapper(net.to(device), device=device, classes=classes)

    logger.info("Model loaded and wrapped.")

   #### Uncertainty evaluation

    # Count per-class pixel totals once; GT labels are invariant under image transforms.
    # Sizes each UQ buffer: N_C = ceil(total_pix[C] * cal_fraction * margin).
    CAL_FRACTION = 0.2
    MARGIN = 1.5
    count_loader = get_ACDC_dataloader(
        root_dir=dataset_path, batch_size=BATCH_SIZE, target_size=(256, 256),
        shuffle=False, num_workers=0, transform=None,
    )
    total_pix_all = np.zeros(len(model.classes_), dtype=np.int64)
    for _, count_gt in count_loader:
        gt_pix = flatten_batch(count_gt).ravel().numpy().astype(int)
        for C in model.classes_:
            total_pix_all[C] += (gt_pix == C).sum()
    logger.info("Per-class pixel totals: %s", total_pix_all)

    uqs = []
    for C in model.classes_:
        N_C = math.ceil(int(total_pix_all[C]) * CAL_FRACTION * MARGIN)
        uqs.append(UncertaintyQuantifier(N=N_C, classes=[C]))
        logger.info("UQ class %d: N=%d", C, N_C)

    iter_coverages_ = []
    iter_set_sizes_ = []
    iter_accuracies_ = []
    iter_alphas_ = []
    iter_alphas_std_ = []
    iter_uncertainties_ = []
    iter_uncertainties_std_ = []

    noises = np.arange(0, 0.8, 0.15)      # AWGN
    # noises = np.arange(0, 1.0, 0.15)    # RandomPerspective
    # noises = np.arange(0, 500.0, 250)   # ElasticTransform


    # Dataframe with the results:
    # +
    level0_cols = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    level1_cols = ['U', r'$\sigma_U$', 'U_E', r'$\sigma_{U_E}$']

    column_index = pd.MultiIndex.from_product([level0_cols, level1_cols],
                                              names=['Classes', 'Uncertainty'])

    data_df = pd.DataFrame(index=noises, columns=column_index)
    data_df.index.name = 'Noises'

    iterations = 3
    
    logger.info("Starting tests...")

    for iteration in range(iterations):
        logger.info("\n---------- Iteration %d (of %d)----------\n", iteration+1, iterations)

        coverages:  list[list[float]] = [[] for _ in model.classes_]
        set_sizes:  list[list[float]] = [[] for _ in model.classes_]
        accuracies: list[list[float]] = [[] for _ in model.classes_]

        alphas_tuned:     list[list[float]] = [[] for _ in model.classes_]
        alphas_tuned_std: list[list[float]] = [[] for _ in model.classes_]
        Us_tuned:         list[list[float]] = [[] for _ in model.classes_]
        Us_tuned_std:     list[list[float]] = [[] for _ in model.classes_]

        # Please note that initial alpha values are not reseted for each noise since it is pointless
        #  to start with lower alphas for bigger noise.
        # They are reseted in each iteration.

        for noise in noises:
            logger.info("\n--------------------------------------------------------------------------------"
                            "\nNoise std: %f\n"
                            "--------------------------------------------------------------------------------\n",
                             noise)

            for C in model.classes_:
                uqs[C].reset()

            transform = transforms.Compose([
                MinMaxNormalize(),
                AddGaussianNoise(0., noise),         # AWGN
                # RandomPerspective(noise, 1),       # Random Perspective
                # ElasticTransform(noise),           # Elastic Transform
            ])

            # Accuracy
            logger.info("Checking model empirical uncertainty...")

            whole_data_loader = get_ACDC_dataloader(
                root_dir=dataset_path, batch_size=BATCH_SIZE, target_size=(256, 256),
                shuffle=True, num_workers=0,
                transform=transform,
                )

            correct_pix = np.zeros_like(model.classes_)
            total_pix = np.zeros_like(model.classes_)
            for i, batch in enumerate(whole_data_loader):
                test_img, gt_img = batch
                pred_img = model(test_img)
                gt_pix = flatten_batch(gt_img).cpu().numpy().astype(int)
                pr_pix = flatten_batch(pred_img.cpu().unsqueeze(1)).cpu().numpy().astype(int)
                for C in model.classes_:
                    total_pix[C] += (gt_pix==C).sum()
                    correct_pix[C]+=np.logical_and(pr_pix==C, gt_pix==C).sum()

            for C in model.classes_:
                accuracy = correct_pix[C]/total_pix[C]
                accuracies[C].append(accuracy)
                logger.info("Accuracy for class %d: %f", C, accuracy)

            # Uncertainties
            logger.info("Evaluating model uncertainties...")

            calDataLoader, tuneDataLoader, testDataLoader = get_ACDC_cal_tun_tst_dataloaders(
                root_dir=dataset_path, target_size=(256, 256),
                cal_batch_size=BATCH_SIZE, tune_batch_size=BATCH_SIZE, test_batch_size=BATCH_SIZE,
                cal_transform=transform,
                tune_transform=transform,
                test_transform=transform,
                splits=[0.2, 0.2, 0.6],
                shuffle=True, num_workers=0)

            logger.debug("Length of Calibration set: %d images.", len(calDataLoader.dataset))  #type: ignore
            logger.debug("Length of Tuning set: %d images.", len(tuneDataLoader.dataset))  #type: ignore
            logger.debug("Length of Test set: %d images.", len(testDataLoader.dataset))  #type: ignore

            # Calibration: batch-outer / class-inner (one model forward per batch)
            for X_cal, y_cal in calDataLoader:
                p_cal = model.predict_proba(X_cal)
                y_cal_arr = flatten_batch(y_cal).ravel().numpy().astype(int)
                for C in model.classes_:
                    uqs[C].calibrate_from_proba(p_cal, y_cal_arr, batched=True)

            # Tuning: precompute full tune set, one call per class (not per-batch averaged)
            tune_probs_list, tune_y_list = [], []
            for X_tune, y_tune in tuneDataLoader:
                tune_probs_list.append(model.predict_proba(X_tune))
                tune_y_list.append(flatten_batch(y_tune).ravel().numpy().astype(int))
            tune_probs_all = torch.cat(tune_probs_list, dim=0)
            tune_y_all = np.concatenate(tune_y_list, axis=0)

            U_per_class: dict[int, float] = {}
            alpha_per_class: dict[int, float] = {}
            for C in model.classes_:
                U, alpha = uqs[C].get_uncertainty_from_proba(tune_probs_all, tune_y_all, max_iters=30)
                U_per_class[C] = float(U)
                alpha_per_class[C] = float(alpha)
                logger.info("Class %d tuning: U=%f, alpha=%f", C, U, alpha)
                uqs[C].alpha = U  # INTENTIONAL: alignment test between U and 1-Cov

            # Test: stream per batch, accumulate global coverage per class
            total_test_pix = np.zeros(len(model.classes_), dtype=np.int64)
            covered_test_pix = np.zeros(len(model.classes_), dtype=np.int64)
            max_setsizes_C = np.zeros(len(model.classes_), dtype=np.int64)

            for X_test, y_test in testDataLoader:
                p_test = model.predict_proba(X_test)
                y_test_arr = flatten_batch(y_test).ravel().numpy().astype(int)
                for C in model.classes_:
                    y_p, y_s = uqs[C].predict_from_proba(p_test)
                    mask_C = (y_test_arr == C)
                    n_C = int(mask_C.sum())
                    total_test_pix[C] += n_C
                    if n_C > 0:
                        covered_test_pix[C] += int(y_s[mask_C, C].sum())
                        max_setsizes_C[C] = max(
                            int(max_setsizes_C[C]),
                            int(y_s[mask_C].sum(axis=1).max()),
                        )

            for C in model.classes_:
                U = U_per_class[C]
                alpha = alpha_per_class[C]
                coverage = covered_test_pix[C] / total_test_pix[C] if total_test_pix[C] > 0 else 0.0
                setsize = int(max_setsizes_C[C])

                logger.info("Class %d: alpha=%f, U=%f, coverage=%f", C, alpha, U, coverage)
                coverages[C].append(coverage)
                set_sizes[C].append(setsize)
                alphas_tuned[C].append(alpha)
                alphas_tuned_std[C].append(0.0)
                Us_tuned[C].append(U)
                Us_tuned_std[C].append(0.0)

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
    min_coverages = np.min(iter_coverages, axis=0)
    max_coverages = np.max(iter_coverages, axis=0)
    mean_coverages =  np.mean(iter_coverages, axis=0)
    np.save(model_data_path / Path('mean_coverages.npy'), mean_coverages)
    std_coverages = np.std(iter_coverages, axis=0)
    np.save(model_data_path / Path('std_coverages.npy'), std_coverages)

    iter_set_sizes = np.array(iter_set_sizes_)
    max_set_sizes = np.max(iter_set_sizes, axis=0)
    min_set_sizes = np.min(iter_set_sizes, axis=0)
    mean_set_sizes = np.mean(iter_set_sizes, axis=0)

    iter_accuracies = np.array(iter_accuracies_)
    min_accuracies = np.min(iter_accuracies, axis=0)
    max_accuracies = np.max(iter_accuracies, axis=0)
    mean_accuracies = np.mean(iter_accuracies, axis=0)
    np.save(model_data_path / Path('mean_accuracies.npy'), mean_accuracies)
    std_accuracies = np.std(iter_accuracies, axis=0)
    np.save(model_data_path / Path('std_accuracies.npy'), std_accuracies)

    iter_alphas = np.array(iter_alphas_)
    min_alphas = np.min(iter_alphas, axis=0)
    max_alphas = np.max(iter_alphas, axis=0)
    mean_alphas = np.mean(iter_alphas, axis=0)

    iter_alphas_std = np.array(iter_alphas_std_)
    mean_alphas_std = np.mean(iter_alphas_std, axis=0)

    iter_uncertainties = np.array(iter_uncertainties_)
    min_uncertainties = np.min(iter_uncertainties, axis=0)
    max_uncertainties = np.max(iter_uncertainties, axis=0)
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
        for C in model.classes_:
            data_df.loc[noise, (f'Class {C}','U')] = mean_uncertainties[C][noise == noises]
            data_df.loc[noise, (f'Class {C}',r'$\sigma_U$')] = std_uncertainties[C][noise == noises]
            data_df.loc[noise, (f'Class {C}','U_E')] = 1 - mean_accuracies[C][noise == noises]
            data_df.loc[noise, (f'Class {C}',r'$\sigma_{U_E}$')] = std_accuracies[C][noise == noises]
    data_df.to_pickle(model_tab_path / Path(f'{model_name}_class_uncertainty.pkl'))
    dfs=data_df.style.format('{:.3f}')
    dfs.to_latex(model_tab_path / Path('class_uncertainty.tex'), hrules=True, label=f'tab:{model_name}_class_uncertainty',
                    caption='Class-wise uncertainty and empirical uncertainty for different noise levels.')
    print(data_df)
    data_df.to_latex(tab_path / 'KM_class_uncertainty.tex', index=True, float_format="%.3f")


    if iterations==1:
        U_str = r'${U}$'
        Ue_str = r'${U}_E$'
        Cov_str = r'$1-{Cov}$'
    else:
        U_str = r'$\bar{U}$'
        Ue_str = r'$\bar{U}_E$'
        Cov_str = r'$1-\overline{Cov}$'

    offset = 0.0
    x = np.arange(len(noises))
    x_float = x.astype(float)
    xlabels = [f'{noise:.2f}' for noise in noises]


    fig, axs = plt.subplots(1, 4, figsize=(7, 7/4), layout='constrained')
    axs = axs.flatten()

    for i in range(len(model.classes_)):
        ax = axs[i]
        ax.set_title(f'Class {i}')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0.1))

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

    output_img_path = Path(img_path / model_name)
    output_img_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_img_path / f'ACDC_example_it{iterations}.pdf')


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

    main(img_path=img_path, tab_path=tab_path, data_path=data_path, transform=transform)
    plt.show()
