import logging
import os

import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from utrace import UncertaintyQuantifier

logger = logging.getLogger(__name__)


def get_model(name: str, random_state: int):
    """INstantiates the base regression models."""
    if name == "RF":
        return RandomForestRegressor(n_estimators = 100, random_state = random_state)
    elif name == "MLP":
        return MLPRegressor(hidden_layer_sizes = (64, 64), max_iter = 500,
                            random_state = random_state)
    else:
        raise ValueError(f"Unknown mdoel {name}")

def process_single_split(n_iter, X_ref, y_ref, black_box_model, eps_values,
                         target_ratios, model_name):
    """Executes a single Monte Carlo split Using UncertaintyQuantifier in
    regression mode."""

    results_split = []
    # Split reference set: 40% Calibration, 30% Tuning, 30% Verification
    X_cal, X_aux, y_cal, y_aux = train_test_split(
        X_ref, y_ref, test_size = 0.6, random_state = n_iter
    )
    X_tune, X_ver, y_tune, y_ver = train_test_split(
        X_aux, y_aux, test_size = 0.5, random_state = n_iter 
    )

    # Obtain point predictions from the fittes black box model
    y_hat_cal = black_box_model.predict(X_cal)
    y_hat_tune = black_box_model.predict(X_tune)
    y_hat_ver = black_box_model.predict(X_ver)

    # Calibrate U-TraCE on regression absolute errors
    uq = UncertaintyQuantifier(N = len(y_cal), score = 'abs_error')
    uq.calibrate(y_hat_cal, y_cal)

    # Grid search over tolerance epsilon an target ratios
    for eps in eps_values:
        # Out-of-tolerance fraction in verfication set $U_E$
        out_of_tol_ver = np.abs(y_ver-y_hat_ver) > eps
        U_E = np.mean(out_of_tol_ver)

        for eta in target_ratios:
            # Estimate risk level (U_star) and optimal alpha via JAX search
            U_star, opt_alpha = uq.get_uncertainty(
                softmax = y_hat_tune, y = y_tune,
                eps = eps, target_ratio = eta)

            '''
            min_alpha = 1.0/len(y_cal)
            if np.isnan(opt_alpha) or opt_alpha < min_alpha:
                continue
            '''   
            # Evaluate bounds on verification set with optimal alpha
            uq.alpha = opt_alpha
            lows, ups = uq.predict(y_hat_ver)

            # Empirical metrics
            cov = np.mean((y_ver >= lows) & (y_ver <= ups))
            avg_L = np.mean(ups - lows)

            """
            # Conditional coverage inside out-of-tolerance region
            if np.sum(out_of_tol_ver) > 0:
                 ratio_real = np.mean((y_ver[out_of_tol_ver] >= lows[out_of_tol_ver]) &
                                        (y_ver[out_of_tol_ver] <= ups[out_of_tol_ver])) 
            else:
                ratio_real = 1.0
            """

            results_split.append({
                'Model': model_name, 'Split': n_iter, 'Epsilon': eps,
                'Target Ratio': eta, 'Optimal Alpha': opt_alpha,
                'L': avg_L,
                'Coverage': cov,
                'U_E': U_E, 'U_star': U_star,
                'Lower Bounds': lows, 'Upper Bounds': ups
            })

    return results_split

def run_single_model_experiment(X, y, model_name, black_box_model, eps_fractions,
                                targets_ratios, L_splits = 500, random_state = 42,
                                log_every: int = 50) -> pd.DataFrame:
    """
    Runs Monte Carlo splits sequentially for a single fitted black box model.
    """

    # Split data set into training (50%) and reference set (50%).
    logger.info("Splitting dataset into training and reference sets ...")

    X_train, X_ref, y_train, y_ref = train_test_split(
        X, y, train_size = 0.5, random_state = random_state
    )

    # Compute absolute target scale for epsilon tolerances
    sigma_y_train = np.std(y_train)
    eps_values = [frac*sigma_y_train for frac in eps_fractions]

    # Fit black box model on training set
    logger.info("Fitting model %s on training set ...", model_name)
    black_box_model.fit(X_train, y_train)

    results = []
    logger.info("Starting Monte Carlo evaluation (%d splits) ...", L_splits)
    # Sequential split evaluation loop
    for kk in range(L_splits):
        if (kk+1) % log_every == 0 or kk == 0 or (kk+1)==L_splits:
            logger.info("Model %s | Processing split %d/%d", model_name, kk+1, L_splits)

        split_results = process_single_split(
            n_iter = kk, X_ref = X_ref, y_ref = y_ref,
            black_box_model = black_box_model,
            eps_values = eps_values,
            target_ratios = targets_ratios,
            model_name = model_name)
        results.extend(split_results)

    df_results = pd.DataFrame(results)
    df_results['Sigma_Y'] = sigma_y_train
    return df_results

def run(dataset_dict: dict, model_names: dict, eps_fractions: np.ndarray,
        target_ratios: np.ndarray, output_dir: str = "./data/regression", 
        save_results: bool = True,
        L_splits: int = 500, random_state: int = 42,
        log_every: int = 50) -> pd.DataFrame:

    if save_results and output_dir:
        os.makedirs(output_dir, exist_ok = True)

    all_experiments = []

    for d_name, (X, y) in dataset_dict.items():
        logger.info("\nProcessing Dataset: %s", d_name)

        for m_name, black_box_model in model_names.items():
            logger.info("\n Running Model %s", m_name)
            try:
                df_res = run_single_model_experiment(
                    X = X, y = y, model_name = m_name,
                    black_box_model = black_box_model,
                    eps_fractions = eps_fractions,
                    targets_ratios = target_ratios,
                    L_splits = L_splits,
                    random_state = random_state,
                    log_every=log_every 
                )
                df_res["dataset"] = d_name
                all_experiments.append(df_res)
                if save_results and output_dir:
                    filename = f"{d_name}_{m_name}_utrace.csv"
                    filepath = os.path.join(output_dir, filename)
                    df_res.to_csv(filepath, index = False)

            except Exception as e:
                logger.error("Failed on datast %s with model %s: %s", 
                             d_name, m_name, e, exc_info = True)
                continue

    if all_experiments:
        master_df = pd.concat(all_experiments, ignore_index = True)
        return master_df
    else:
        logger.warning("No experiments completed successfully.")
        return pd.DataFrame()


if __name__ == "__main__":

    import utrace.utils.regression_data.datasets as data

    logging.basicConfig(level = logging.INFO,
                        format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    dataset = {
        "Synthetic_Gaussian": data.simulate_gaussian(n = 10000, sigma = 0.9,
                                                     random_state = 42)
    }

    models = {
        "MLP": get_model("MLP", random_state = 42)
    }

    eps_fracs = np.array([0.1, 0.25, 0.5, 0.75, 1.00])
    target_eta = np.array([0.9, 1.0])
    master_results = run(dataset_dict = dataset, model_names = models,
                         eps_fractions = eps_fracs,
                         target_ratios = target_eta)

    import matplotlib.pyplot as plt

    df = master_results.copy()
    df['Normalized_Epsilon'] = df['Epsilon']/df['Sigma_Y']

    print(df['Epsilon'].unique())

    #df = df[df['Target Ratio'].round(5).isin(target_eta)]

    group_U = df.groupby(['Normalized_Epsilon', 'Target Ratio']).agg(
        U_E_mean = ('U_E', 'mean'), U_star_mean = ('U_star', 'mean')
    ).reset_index()

    fig, axs = plt.subplots()
    axs.plot(group_U['Normalized_Epsilon'], group_U['U_E_mean'], color = 'blue',
             marker = 'o', label = r'Empirical Error ($U_E$)')
    for tr in target_eta:
        sub_tr = group_U[np.isclose(group_U['Target Ratio'], tr)]
        axs.plot(sub_tr['Normalized_Epsilon'], sub_tr['U_star_mean'],
                 linestyle = '--', marker = 's',
                 label = rf'$U^\star$ for $\bar{{\eta}} = {tr:0.1f}$')
    axs.set_xlabel(r'Normalized tolerance ($\varepsilon/\sigma_Y$)')
    axs.legend()
    plt.show()

    group_opt_tr = df.groupby(['Normalized_Epsilon', 'Target Ratio']).agg(
        opt_alpha_mean = ('Optimal Alpha', 'mean')).reset_index()

    fig2, axs2 = plt.subplots()
    for tr in target_eta:
        sub = group_opt_tr[np.isclose(group_opt_tr['Target Ratio'], tr)]
        print(sub['opt_alpha_mean'])
        axs2.plot(sub['Normalized_Epsilon'], sub['opt_alpha_mean'], 
                 linestyle = '-', marker = 'o',
                 label = rf'$\bar{{\eta}}={tr:.2f}$')

    axs2.set_xlabel(r'Normalized tolerance ($\varepsilon/\sigma_Y$)')
    axs2.set_ylabel(r'Optimal alpha ($\alpha^\star$)')
    axs2.legend()
    plt.show()

    