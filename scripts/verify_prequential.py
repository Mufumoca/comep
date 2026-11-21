from collections import deque

import numpy as np

from comep.policy import FixedBankPolicy
from comep.prequential import PrequentialPolicy, forecast_loss, paired_gate


def main():
    assert not paired_gate(deque([(.1, False)]*199))["deploy_joint"]
    assert paired_gate(deque([(.1, False)]*200))["deploy_joint"]
    assert not paired_gate(deque([(-.1, True)]*400))["deploy_joint"]
    np.testing.assert_allclose(forecast_loss(.5, 1), np.log(2))
    for mode, legacy in (("additive", "normalized-additive"), ("joint", "normalized-joint-soft")):
        model = PrequentialPolicy(mode, [8]*8)
        baseline = FixedBankPolicy(legacy, [8]*8, window=4000, refit_every=100, ridge=1.)
        random = np.random.default_rng(17)
        for episode in range(320):
            context = 2*(episode % 2)-1
            gates = np.where(np.arange(8)%2 == episode % 2, 1., .25)
            chosen, _ = model.choose(context, gates)
            assert chosen == baseline.choose(context, gates)[0]
            reward = int(random.random() < .4)
            model.observe(chosen, reward)
            baseline.observe(context, gates, chosen, reward)
    model = PrequentialPolicy("gated", [8, 8])
    for episode in range(600):
        chosen, metadata = model.choose(1, np.ones(2), episode % 4)
        if episode == 0:
            try:
                model.choose(1, np.ones(2))
            except RuntimeError:
                pass
            else:
                raise AssertionError("unconsumed pending action accepted")
        reward = int(chosen in (1, 2))
        observed = model.observe(chosen, reward)
        if episode < 100:
            assert observed["paired_logloss_gain"] is None
        else:
            expected = forecast_loss(metadata["forecasts"]["additive"], reward)-forecast_loss(metadata["forecasts"]["joint"], reward)
            np.testing.assert_allclose(observed["paired_logloss_gain"], expected)
        if episode == 99:
            assert len(model.loss_history) == 0
        if episode == 299:
            assert len(model.loss_history) == 200
    assert len(model.loss_history) == 400
    assert model.gate["deploy_joint"]
    print("PREQUENTIAL_WITNESS baseline equivalence, pending action, prior forecasts, first-fit boundary, gate and positive fixture passed")


if __name__ == "__main__":
    main()
