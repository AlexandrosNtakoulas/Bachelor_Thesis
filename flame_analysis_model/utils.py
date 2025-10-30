# ----------------------------------------------------
# Utilities
# ----------------------------------------------------
def _unwrap_scalar(x):
    return x[0] if isinstance(x, list) else x
