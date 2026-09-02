.
├── BACKLOG.md
├── CLAUDE.md
├── CONTRIBUTING.md
├── FINDINGS.md
├── git_log.log
├── LICENSE.md
├── MIGRATION.md
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
│       ├── tests
│       │   ├── conftest.py
│       │   ├── __init__.py
│       │   └── test_utils.py
│       ├── uncertaintyQuantifier.py
│       └── utils
│           ├── __init__.py
│           ├── onnx
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
│   │   ├── test_deferred_sort_buffer.py
│   │   ├── test_import_properties.py
│   │   ├── test_label_dtype_canonicalisation.py
│   │   ├── test_masked_quantile.py
│   │   ├── test_max_n_overflow.py
│   │   ├── test_score_param_validation.py
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
│           └── test_torch_label_input.py
└── uv.lock

14 directories, 61 files
