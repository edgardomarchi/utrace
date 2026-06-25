import logging
from pathlib import Path


def setup_example_io(script_path, *, transform: str | None = None,
                     roots=("img", "data", "tab"), level: int = logging.DEBUG):
    """Create the standard output roots for an example script and configure logging.

    Paths computed (where stem = Path(script_path).stem):
        img/<stem>/[<transform>/]
        data/<stem>/[<transform>/]
        tab/<stem>/[<transform>/]
    When transform is None, the <transform> level is omitted.

    Only the roots named in `roots` are created on disk (mkdir). All three paths
    are always computed and returned regardless. Default ("img", "data", "tab")
    creates all three, preserving prior behavior. The log dir + log_path are always
    set up unconditionally (independent of `roots`).

    Logging is ALWAYS flat: log/<stem>.log (no transform/model level).

    Logging config (root logger):
        - root level = `level`
        - console handler: INFO, format "[%(levelname)s]: %(message)s"
        - file handler (only if root level <= DEBUG): log/<stem>.log, mode='w', DEBUG,
          format "[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)20s()]: %(message)s"
        - third-party loggers 'matplotlib' and 'jax' set to WARNING

    Returns
    -------
    (img_dir, data_dir, tab_dir, log_path) : tuple[Path, Path, Path, Path]
        All four paths always returned. Only the roots listed in `roots` are
        actually created on disk.
    """
    stem = Path(script_path).stem

    def _root(name):
        path = Path(name) / stem
        if transform is not None:
            path = path / transform
        if name in roots:
            path.mkdir(parents=True, exist_ok=True)
        return path

    img_dir = _root("img")
    data_dir = _root("data")
    tab_dir = _root("tab")

    Path("log").mkdir(parents=True, exist_ok=True)
    log_path = Path("log") / f"{stem}.log"

    logger = logging.getLogger('')
    logger.handlers.clear()
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s]: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if logger.level <= logging.DEBUG:
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)20s()]: %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
    jax_logger = logging.getLogger('jax')
    jax_logger.setLevel(logging.WARNING)

    return img_dir, data_dir, tab_dir, log_path
