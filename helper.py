def short_name(name):
    return name.split('.')[5]

def type_from_feature_name(name):
    return name.split('.')[4]

def short_name_with_type(name):
    return name.replace('ch.uzh.ciclassifier.features.','')