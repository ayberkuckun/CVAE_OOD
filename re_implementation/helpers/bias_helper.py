def get_bias_correction_term(dist):
    if dist == "cBern":
        return 1
    elif dist == "cat":
        return 1
