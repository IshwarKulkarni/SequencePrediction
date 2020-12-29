import importlib


def make_class_from_module(full_class_name):
    """Given a fully qualifies class name with module, return the class object"""
    assert '.' in full_class_name, 'Passed in class name is not fully qualified'
    names = full_class_name.split('.')
    class_name = names[-1]
    module_name = '.'.join(names[0:-1])

    module = importlib.import_module(module_name)
    return getattr(module, class_name)
