try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version(__name__.split('.', maxsplit=1)[0])
    except PackageNotFoundError:
        # package is not installed
        pass
    finally:
        del version, PackageNotFoundError
except ModuleNotFoundError:
    from pkg_resources import DistributionNotFound, get_distribution

    try:
        __version__ = get_distribution(
            __name__.split('.', maxsplit=1)[0]
        ).version
    except DistributionNotFound:
        # package is not installed
        pass
    finally:
        del get_distribution, DistributionNotFound
