import os


def delete_log():
    if os.path.exists('pyfibre.log'):
        os.remove('pyfibre.log')
