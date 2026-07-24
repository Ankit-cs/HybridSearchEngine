def to_fixed_point(score: float, frac_bits: int = 16) -> int:
    """
    Converts a floating point score to a fixed-point integer.
    Useful for ensuring bitwise-deterministic sorting across different hardware architectures.
    
    Args:
        score: The floating point score to convert
        frac_bits: Number of bits to use for the fractional part (default: 16 for Q16.16 equivalent scaling)
        
    Returns:
        The integer representation of the fixed-point value
    """
    return int(round(score * (1 << frac_bits)))


def sort_by_fixed_point(items: list, key_index: int = 1, reverse: bool = True, frac_bits: int = 16) -> list:
    """
    Sorts a list of tuples using a fixed-point conversion on a specific tuple index.
    
    Args:
        items: List of items to sort (usually a list of tuples like [(doc_id, score), ...])
        key_index: The index within the tuple to sort by (default: 1, assuming score is at index 1)
        reverse: Whether to sort in descending order (default: True, for highest relevance first)
        frac_bits: Fixed point precision (default: 16)
        
    Returns:
        The sorted list
    """
    return sorted(items, key=lambda x: to_fixed_point(x[key_index], frac_bits), reverse=reverse)
