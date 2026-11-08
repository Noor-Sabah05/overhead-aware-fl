"""Controllers cannot inspect test accuracy or the true background-load phase."""
import math
import time
import numpy as np


class ReactiveController:
    """A deliberately simple epoch-throughput baseline, without probe overhead."""
    def __init__(self, options, baseline):
        self.options = options
        self.index = len(options) - 1
        self.baseline = baseline
        self.bad = self.good = 0
        self.trial = None

    @property
    def threads(self):
        return self.options[self.index]

    def observe(self, throughput):
        event = 'hold'
        if self.trial is not None:
            old_index, old_throughput = self.trial
            if throughput < old_throughput * 1.05:
                self.index = old_index
                event = 'revert'
            else:
                event = 'accept'
            self.trial = None
            self.bad = self.good = 0
        elif throughput < .9 * self.baseline[str(self.threads)]:
            self.bad += 1
            self.good = 0
            if self.bad >= 2 and self.index > 0:
                self.trial = (self.index, throughput)
                self.index -= 1
                self.bad = 0
                event = 'try_lower'
        else:
            self.good += 1
            self.bad = 0
            if self.good >= 2 and self.index < len(self.options) - 1:
                self.index += 1
                self.good = 0
                event = 'try_higher'
        return {'event': event, 'next_threads': self.threads}


def gate(incumbent_s, candidate_s, probe_wall_s, remaining_epochs, cfg, gated=True):
    """Pure decision rule; times are matched probe seconds per training example."""
    gain = 1 - candidate_s / incumbent_s
    horizon = min(cfg['horizon_epochs'], remaining_epochs)
    savings = (incumbent_s - candidate_s) * cfg['client_examples'] * horizon
    accepted = gain >= cfg['min_gain'] and (not gated or savings > cfg['payback_factor'] * probe_wall_s)
    return {'accepted': bool(accepted), 'gain_fraction': gain, 'horizon_epochs': horizon,
            'predicted_savings_s': savings, 'required_savings_s': cfg['payback_factor'] * probe_wall_s,
            'reason': 'accept' if accepted else 'insufficient_gain' if gain < cfg['min_gain'] else 'not_amortized'}


class ProbeController:
    """Matched AB/BA probes + an explicit measured-overhead payback gate."""
    def __init__(self, options, initial, cfg, gated=True):
        self.options, self.threads, self.cfg, self.gated = options, initial, cfg, gated
        self.probe_number = 0

    def choose(self, pool, state, optimizer, indices, local_epoch, remaining_epochs):
        if local_epoch % self.cfg['probe_every'] or len(self.options) == 1:
            return self.threads, {'event': 'cooldown'}
        i = self.options.index(self.threads)
        neighbors = self.options[max(0, i-1):i] + self.options[i+1:i+2]
        candidate = neighbors[self.probe_number % len(neighbors)]
        current = self.threads
        probe_ids = indices[:self.cfg['probe_batches'] * self.cfg['batch']]
        # Each call reloads the SAME parameters AND optimizer state, and uses SAME indices.
        # A B B A counterbalances order within each decision, not just across decisions.
        order = [current, candidate, candidate, current]
        if self.probe_number % 2:
            order = [candidate, current, current, candidate]
        measurements = {current: [], candidate: []}
        start = time.perf_counter()
        for threads in order:
            result = pool.call(threads, 'probe', state, probe_ids, optimizer)
            measurements[threads].append(result['train_s'] / result['examples'])
        wall = time.perf_counter() - start
        means = {t: float(np.mean(v)) for t, v in measurements.items()}
        cfg = {**self.cfg, 'client_examples': len(indices)}
        decision = gate(means[current], means[candidate], wall, remaining_epochs, cfg, self.gated)
        if decision['accepted']:
            self.threads = candidate
        self.probe_number += 1
        return self.threads, dict(event='probe', old_threads=current, candidate_threads=candidate,
            next_threads=self.threads, probe_wall_s=wall, probe_examples=len(probe_ids)*4,
            seconds_per_example={str(t): v for t, v in measurements.items()}, order=order, **decision)
