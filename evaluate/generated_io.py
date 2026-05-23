"""Generated-sample JSON naming (must match notebooks/ca/generate.ipynb)."""


def condition_filename(trend: float, rv: float, ir: float, vix: float) -> str:
    return f"t{int(trend)}r{int(rv)}i{int(round(ir * 100))}v{int(vix)}.json"
