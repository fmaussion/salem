try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version(__name__.split('.', maxsplit=1)[0])
    except PackageNotFoundError:
        # package is not installed
        pass
    finally:
        del version, PackageNotFoundError
except ModuleNotFoundError:
    from pkg_resources import get_distribution, DistributionNotFound
    try:
        __version__ = get_distribution(__name__.split('.', maxsplit=1)[0]).version
    except DistributionNotFound:
        # package is not installed
        pass
    finally:
        del get_distribution, DistributionNotFound
