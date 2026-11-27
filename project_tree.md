.
├── BACKLOG.md
├── CHANGELOG.md
├── CLAUDE.md
├── CONTRIBUTING.md
├── FINDINGS.md
├── .github
│   ├── dependabot.yml
│   └── workflows
│       └── tests.yml
├── .gitignore
├── LICENSE.md
├── MIGRATION.md
├── notebooks
│   ├── acdc_bm_cache.py
│   ├── acdc_periodic_verification.py
│   ├── mnist_method_evaluation.py
│   ├── model_comparison.py
│   ├── periodic_verification_demo.py
│   └── run_notebook.py
├── project_tree.md
├── pyproject.toml
├── README.md
├── scripts
│   ├── ACDC_example.py
│   ├── btorch_MNIST_test.py
│   ├── _common.py
│   ├── convergence_analysis.py
│   ├── data_size_analysis.py
│   ├── MNIST_class_conditional_example.py
│   ├── MNIST_example.py
│   ├── MNIST_test_convergence.py
│   ├── MNIST_test_coverage.py
│   ├── regression_example.py
│   └── setsize_analysis.py
├── src
│   └── utrace
│       ├── bounds.py
│       ├── budget.py
│       ├── cli.py
│       ├── __init__.py
│       ├── __main__.py
│       ├── scores
│       │   ├── __init__.py
│       │   ├── jax_impl.py
│       │   └── objects.py
│       ├── uncertaintyQuantifier.py
│       └── utils
│           ├── __init__.py
│           ├── pytorch
│           │   ├── dataset_wrapper.py
│           │   ├── example_models.py
│           │   ├── helpers.py
│           │   ├── __init__.py
│           │   ├── model_wrapper.py
│           │   └── transforms.py
│           ├── regression_data
│           │   ├── datasets.py
│           │   └── __init__.py
│           ├── tensors.py
│           ├── utils_jax.py
│           └── utils.py
├── tests
│   ├── core
│   │   ├── test_alpha_setter_quantile_equiv.py
│   │   ├── test_api.py
│   │   ├── test_betting_montecarlo.py
│   │   ├── test_betting.py
│   │   ├── test_bound_objects.py
│   │   ├── test_budget_units.py
│   │   ├── test_buffer_sort_shape_stability.py
│   │   ├── test_calibrate_device_reconciliation.py
│   │   ├── test_calibrate_jit_marginal.py
│   │   ├── test_confidence_bound_montecarlo.py
│   │   ├── test_confidence_bounds.py
│   │   ├── test_deferred_sort_buffer.py
│   │   ├── test_ec_empirical_bernstein.py
│   │   ├── test_eval_set_device_reconciliation.py
│   │   ├── test_eval_set_split.py
│   │   ├── test_get_uncertainty_device_reconciliation.py
│   │   ├── test_get_uncertainty_single_dataset_pinned.py
│   │   ├── test_import_properties.py
│   │   ├── test_label_dtype_canonicalisation.py
│   │   ├── test_label_dtype_policy.py
│   │   ├── test_label_validation.py
│   │   ├── test_masked_quantile.py
│   │   ├── test_max_n_overflow.py
│   │   ├── test_per_sample_path.py
│   │   ├── test_regression_confidence_bounds_montecarlo.py
│   │   ├── test_regression_confidence_bounds.py
│   │   ├── test_regression_task.py
│   │   ├── test_scipy_import_is_lazy.py
│   │   ├── test_score_accepted.py
│   │   ├── test_score_objects.py
│   │   ├── test_score_param_validation.py
│   │   ├── test_search_status.py
│   │   ├── test_search_uncertainty.py
│   │   ├── test_single_estimand_montecarlo.py
│   │   ├── test_single_estimand.py
│   │   └── test_uncertainty_budget.py
│   └── integration
│       └── torch
│           ├── baselines
│           │   ├── mean_alphas.npy
│           │   ├── mean_coverages.npy
│           │   ├── mean_uncertainties.npy
│           │   └── README.md
│           ├── _baselines.py
│           ├── conftest.py
│           ├── regenerate_baselines.py
│           ├── test_golden_mnist.py
│           ├── test_search_max_iters_stability.py
│           └── test_torch_label_input.py
└── uv.lock

16 directories, 98 files
