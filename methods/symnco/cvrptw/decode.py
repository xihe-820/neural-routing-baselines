"""Strict decoder for the standalone SymNCO action representation."""


def decode_selected_action(action, problem_size):
    if not isinstance(action, list) or len(action) < 3:
        raise ValueError("SymNCO selected action must be a nonempty list")
    if any(isinstance(node, bool) or not isinstance(node, int) or
           node < 0 or node > problem_size for node in action):
        raise ValueError("SymNCO selected action contains an invalid node")
    if action[0] != 0 or action[-1] != 0:
        raise ValueError("SymNCO selected action must be explicitly depot-closed")
    return list(action)
