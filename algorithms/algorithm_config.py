import algorithms

algorithm_mapping = {
    "WCA": {
        "module": algorithms.WCA.wca,
        "config": {"LB": -5, "UB": 5, "nvars": 2, "npop": 50, "nsr": 4, "dmax": 1e-16, "max_it": 100}
    },
    "TGA": {
        "module": algorithms.TGA.tga,
        "config": {"LB": -5, "UB": 5, "nvars": 2, "npop": 100, "N1": 40, "N2": 40, "N3": 20, "N4": 30, "lambda": 0.5, "theta": 1.1, "max_it": 100}
    },
    "MBO": {
        "module": algorithms.MBO.mbo,
        "config": {"LB": -5, "UB": 5, "nvars": 2, "npop": 100, "Keep": 2, "p": 0.4167, "period": 1.2, "smax": 1.0, "BAR": 0.4167, "max_it": 100}
    }
}