import importlib
import os
from collections import OrderedDict


def create_default_local_file():
    path = os.path.join(os.path.dirname(__file__), 'local.py')

    empty_str = '\'\''
    default_settings = OrderedDict({
        'workspace_dir': empty_str,
        'pretrained_networks': 'self.workspace_dir + \'/pretrained_networks/\'',
        'isogd_dir': empty_str,
        'thu_dir': empty_str,
        'nvgesture_dir': empty_str,
        'ijcai_miga_track1_dir': empty_str,
        'imigue_rgb_trainval_root': empty_str,
        'imigue_sk_trainval_root': empty_str,
        'imigue_rgb_test_root': empty_str,
        'imigue_sk_test_root': empty_str})

    comment = {'workspace_dir': 'Base directory for saving network checkpoints.'}

    with open(path, 'w') as f:
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self):\n')

        for attr, attr_val in default_settings.items():
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            if comment_str is None:
                f.write('        self.{} = {}\n'.format(attr, attr_val))
            else:
                f.write('        self.{} = {}    # {}\n'.format(attr, attr_val, comment_str))


def create_default_local_file_ITP_train(workspace_dir, data_dir):
    path = os.path.join(os.path.dirname(__file__), 'local.py')

    empty_str = '\'\''
    default_settings = OrderedDict({
        'workspace_dir': workspace_dir,
        'pretrained_networks': os.path.join(workspace_dir, 'pretrained_networks'),
        'icpr_dir': os.path.join(data_dir, 'ICPR_MMVPR_Track3'),
        'isogd_dir': os.path.join(data_dir, 'IsoGD'),
        'thu_dir': os.path.join(data_dir, 'THU-READ'),
        'nvgesture_dir':os.path.join(data_dir, 'NvGesture'),
        'ijcai_miga_track1_dir': os.path.join(data_dir, 'iMiGUE'),
        'imigue_rgb_trainval_root': data_dir,
        'imigue_sk_trainval_root': data_dir,
        'imigue_rgb_test_root': empty_str,
        'imigue_sk_test_root': empty_str})

    comment = {'workspace_dir': 'Base directory for saving network checkpoints.'}

    with open(path, 'w') as f:
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self):\n')

        for attr, attr_val in default_settings.items():
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            if comment_str is None:
                if attr_val == empty_str:
                    f.write('        self.{} = {}\n'.format(attr, attr_val))
                else:
                    f.write('        self.{} = \'{}\'\n'.format(attr, attr_val))
            else:
                f.write('        self.{} = \'{}\'    # {}\n'.format(attr, attr_val, comment_str))


def env_settings():
    env_module_name = 'lib.train.admin.local'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.EnvironmentSettings()
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local.py')

        create_default_local_file()
        raise RuntimeError(
            'YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all the paths you need. Then try to run again.'.format(
                env_file))
