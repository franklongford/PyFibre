TASKS = 'envisage.ui.tasks.tasks'

MULTI_IMAGE_FACTORIES = 'pyfibre.core.multi_image_factories'


def plugin_id(name, version):
    """Creates an ID for the plugins.

    Parameters
    ----------
    name: str
        A string identifying the plugin.
    version: int
        A version number for the plugin.
    """
    if not isinstance(version, int) or version < 0:
        raise ValueError("version must be a non negative integer")

    return '.'.join(["pyfibre", "plugin", name, str(version)])
