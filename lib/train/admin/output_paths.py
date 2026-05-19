import os


def get_output_area(script_name):
    if script_name == 'sk_vit':
        return 'rgb'
    if script_name == 'sk_maga':
        return 'sk'
    return script_name


def resolve_save_dir(save_dir, script_name):
    area = get_output_area(script_name)
    abs_save_dir = os.path.abspath(save_dir)
    if os.path.basename(abs_save_dir) == area:
        return abs_save_dir
    return os.path.join(abs_save_dir, area)
