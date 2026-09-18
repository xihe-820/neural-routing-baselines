from common.cvrptw_formal import canonicalize_official_actions


def decode_selected_action(action, problem_size):
    return canonicalize_official_actions(action, problem_size)


class ActionCapture:
    """Capture the exact action passed to one official env reward call."""
    def __init__(self, env):
        self.env, self.actions, self._original = env, [], None
        self._had_instance_attribute = False
        self._instance_attribute = None

    def __enter__(self):
        self._had_instance_attribute = "get_reward" in vars(self.env)
        self._instance_attribute = vars(self.env).get("get_reward")
        self._original = self.env.get_reward

        def capture(td, actions):
            self.actions.append(actions.detach())
            return self._original(td, actions)

        self.env.get_reward = capture
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._had_instance_attribute:
            self.env.get_reward = self._instance_attribute
        else:
            delattr(self.env, "get_reward")
        return False
