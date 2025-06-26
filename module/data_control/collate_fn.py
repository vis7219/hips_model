def collate_fn(batch):
    """
    Collate function for the DataLoader
    """
    proteins, ligands, targets = zip(*batch)
    return list(proteins), list(ligands), list(targets)