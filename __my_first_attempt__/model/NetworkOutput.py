from typing import List, NamedTuple


class NetworkOutput(NamedTuple):
    # output format of the model
    value: float
    value_prefix: float
    policy_logits: List[float]
    hidden_state: List[float]
    reward_hidden: object