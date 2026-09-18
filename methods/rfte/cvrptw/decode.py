from common.cvrptw_formal import canonicalize_official_actions


def decode_selected_action(action, problem_size):
    return canonicalize_official_actions(action, problem_size)
