

def get_base_new_classes(dataset):
    '''
    Returns the base and new classes for the given dataset.
    '''
    
    if dataset == 'simd':
        base_classes = ['car', 'helicopter', 'boat', 'long-vehicle']
        new_classes = ['trainer-aircraft', 'pushback-truck', 'propeller-aircraft', 'truck',
                        'charted-aircraft', 'figther-aircraft', 'van', 'airliner', 'stair-truck', 'bus']
    elif dataset == 'dior':
        base_classes = ['airplane', 'baseballfield', 'basketballcourt', 'groundtrackfield', 'harbor', 'ship', 'tenniscourt', 'storagetank']
        new_classes = ['airport', 'bridge', 'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'overpass', 'stadium', 'windmill', 'trainstation', 'vehicle']
    
    return base_classes, new_classes