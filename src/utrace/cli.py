"""A sample CLI."""

import argparse
import logging

logger = logging.getLogger(__name__)

def cli():
    parser = argparse.ArgumentParser(description='Run examples for UQM')

    parser.add_argument('-e', '--example', type=str, choices=['convergence','noise'], default='convergence',
                        help='Example tu run: convergence or noise sweep')
    parser.add_argument('-d', '--dataset', type=str, default='../data/ACDC',
                        help='Dataset path (for now, only expects ACDC dataset)')

    # parser.add_argument('-i', '--info', help='Show current configuration', action='store_true')

    parser.add_argument('-l', '--log', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')

    args = parser.parse_args()
    # if args.info:
    #     print('Current configuration:')
    #     return

    # Set the root logger level based on the -l argument
    log_level = getattr(logging, args.log.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {args.log}")
    logging.getLogger('').setLevel(log_level)

if __name__ == '__main__':  # pragma: no cover
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
        file_handler = logging.FileHandler('log/debug.log', mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    cli()  # pylint: disable=no-value-for-parameter
