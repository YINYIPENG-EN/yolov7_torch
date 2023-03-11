import yaml
# def get_classes(classes_path):
#     with open(classes_path, encoding='utf-8') as f:
#         class_names = f.readlines()
#     class_names = [c.strip() for c in class_names]
#     return class_names, len(class_names)
def get_classes(classes_path):
    with open(classes_path,encoding='utf-8') as f:
        yaml_file = yaml.safe_load(f)
        class_names = yaml_file['names']
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)