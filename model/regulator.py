
def regulate(raw_phonemes, max_allowed):
    """
    ~Regulate~ a series of phonemes by removing those which are infrequent, etc.

    Args:
        raw_phonemes: series of phonemes that includes those which are
            inaccurate, etc.

    Returns:
        list of max_allowed elements encapsulating the "correct" phonemes; if
        the list is not filled, right-pad it with zeros.
    """
    
    
    pass

def filter_sequence(seq, min_combo=2):
    # simple way
    combos = [[k, len(list(g))] for k, g in groupby(seq)]
    nseq = []
    for combo in combos:
        if combo[1] >= min_combo:
            # preserve duplication for repeated filtering
            nseq.extend([combo[0]]*combo[1])
    return nseq

def pad_list(seq, pad_val, max_len):
    return seq + [pad_val] * (max_len - len(seq))
