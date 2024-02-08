

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
    elif dataset == 'fair1m':

        base_classes = ["Small-Car", "Baseball-Field","Basketball-Court", "Football-Field", "Tennis-Court", "Roundabout"]
        new_classes = ["Boeing737", "Boeing777", "Boeing747", "Boeing787", "A320", "A321", "A220", "A330", "A350", "C919",
                        "ARJ21", "other-airplane", "Passenger-Ship", "Motorboat", "Fishing-Boat", "Tugboat", "Engineering-Ship",
                        "Liquid-Cargo-Ship", "Dry-Cargo-Ship", "Warship", "other-ship", "Bus", "Cargo-Truck",
                        "Dump-Truck", "Van", "Trailer", "Tractor", "Truck-Tractor", "Excavator", "other-vehicle", "Intersection", "Bridge"]


        
    
    return base_classes, new_classes