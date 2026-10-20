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
│   └── mnist_method_evaluation.py
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
│   └── setsize_analysis.py
├── src
│   └── utrace
│       ├── cli.py
│       ├── __init__.py
│       ├── __main__.py
│       ├── scores
│       │   ├── __init__.py
│       │   └── jax_impl.py
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
│           ├── tensors.py
│           ├── utils_jax.py
│           └── utils.py
├── tests
│   ├── core
│   │   ├── test_alpha_setter_quantile_equiv.py
│   │   ├── test_api.py
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
│   │   ├── test_scipy_import_is_lazy.py
│   │   ├── test_score_param_validation.py
│   │   ├── test_search_status.py
│   │   └── test_search_uncertainty.py
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

15 directories, 74 files
