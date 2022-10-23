
def build_dynamic_dataset_iter(fields, transform, opts, is_train=True):
    transforms = make_transforms(opts, transform_vcls, fields)