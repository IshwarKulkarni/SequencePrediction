import importlib


def make_class_from_module(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def make_object(module_name, class_name, args):
    c = make_class_from_module(module_name, class_name)
    return c(**args)