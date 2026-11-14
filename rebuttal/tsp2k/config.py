"""Budget-only adaptations; paper registries remain authoritative and unchanged."""
from copy import deepcopy

DATASET = "tsp2000_lkh_500_32.436.pkl"
COUNT = 64
GLOP_SWEEP = {i: [20*i, 25*i, 5*i] for i in range(1, 7)}
UDC_SWEEP = (1, 2, 5, 10, 20, 50)
SIL_SWEEP = (1, 2, 5, 10)
SWEEPS = {"GLOP": tuple(GLOP_SWEEP), "UDC": UDC_SWEEP, "SIL": SIL_SWEEP}
UPSTREAMS = {
    "GLOP": ("henry-yeh/GLOP", "e540bc0153a0598e923e35116deeaecaf9c1cfff"),
    "UDC": ("CIAM-Group/NCO_code", "274df3c4975384592b60fe7f79fbb2441ce11c15"),
    "SIL": ("CIAM-Group/SIL", "9ec783e90a1631f7b95f84eb20f8f9751cb45c10"),
}
BASE_COMMITS = {
    "GLOP": "395070d19ce53c798db629acf70fecdeb18f59b9",
    "UDC": "395070d19ce53c798db629acf70fecdeb18f59b9",
    "SIL": "cc4d017ae00c1ab1c4099e034e6ee819e139d3bd",
}


def solver_config(method, budget):
    if type(budget) is not int or budget not in SWEEPS[method]:
        raise ValueError("budget outside the reviewed sweep")
    if method == "GLOP":
        from methods.glop.paper_protocol import formal_protocol
        config = formal_protocol("TSP", 2000, "official_standard")
        config["revision_iters"] = list(GLOP_SWEEP[budget])
        return config
    if method == "UDC":
        from methods.udc.paper_protocol import budget_config
        config = budget_config("tsp", "fewer")
        config["x"] = budget
        return config
    from methods.sil.config import resolve_config
    config = resolve_config("tsp", 2000, "fewer")
    config.update(budget=budget, budget_label=f"prc{budget}",
                  artifact_class="rebuttal_runtime_quality",
                  paper_result_eligible=False,
                  evaluation_mapping="rebuttal explicit integer PRC budget",
                  budget_mapping_origin="reviewed_rebuttal_sweep")
    return config


def budget_parameter(method):
    return {"GLOP": "revision_iters_multiplier", "UDC": "x_stages",
            "SIL": "prc_budget"}[method]


def protocol(method, budget):
    return {
        "method": method, "problem": "TSP", "problem_size": 2000,
        "dataset_filename": DATASET, "dataset_count": COUNT,
        "budget_parameter": budget_parameter(method), "budget_value": budget,
        "full_solver_config": deepcopy(solver_config(method, budget)),
        "quality_batch_size": 64 if method == "SIL" else 1,
        "timing_batch_size": 1, "training": False,
    }
