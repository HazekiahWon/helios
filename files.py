from common_imports import *
def path_join(*ls_args):
    """

    :param ls_args: no need for a list in form
    :return:
    """
    return os.path.join(*ls_args)

def list_dir(dir, full=True):
    """

    :param dir:
    :param full: if True return full path
    :return: list of paths
    """
    if full:
        return glob.glob(path_join(dir,'*'))
    else: return os.listdir(dir)

def is_dir(name):
    return os.path.isdir(name)

def abs_path(name):
    return os.path.abspath(name)
