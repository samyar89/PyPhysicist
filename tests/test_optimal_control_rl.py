import importlib.util
import sys
from pathlib import Path

import pytest


def _load_rl_module():
    module_path = Path(__file__).resolve().parents[1] / "PyPhysicist" / "optimal_control" / "rl_integration.py"
    spec = importlib.util.spec_from_file_location("rl_integration", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


rl_module = _load_rl_module()
OptimalControlAgent = rl_module.OptimalControlAgent
RLEnvironment = rl_module.RLEnvironment
run_optimal_control_episode = rl_module.run_optimal_control_episode


class ConstantAgent(OptimalControlAgent[float, float]):
    def __init__(self, action: float) -> None:
        self.action = action
        self.observed = []

    def act(self, state: float, time: float) -> float:
        return self.action

    def observe(self, transition):
        self.observed.append(transition)


def test_environment_reset_and_step():
    def dynamics(state: float, action: float, dt: float) -> float:
        return state + action * dt

    def reward_fn(state: float, action: float, next_state: float, time: float) -> float:
        return next_state - state

    def termination_fn(state: float, time: float, step_index: int) -> bool:
        return step_index >= 1

    env = RLEnvironment(
        initial_state=1.0,
        dynamics=dynamics,
        reward_fn=reward_fn,
        termination_fn=termination_fn,
        dt=0.5,
    )

    assert env.reset() == 1.0
    assert env.reset(2.0) == 2.0

    next_state, reward, done = env.step(action=2.0, time=0.0, step_index=0)
    assert next_state == pytest.approx(3.0)
    assert reward == pytest.approx(1.0)
    assert done is False

    next_state, reward, done = env.step(action=2.0, time=0.5, step_index=1)
    assert next_state == pytest.approx(4.0)
    assert reward == pytest.approx(1.0)
    assert done is True


def test_run_optimal_control_episode_tracks_steps():
    def dynamics(state: float, action: float, dt: float) -> float:
        return state + action * dt

    def reward_fn(state: float, action: float, next_state: float, time: float) -> float:
        return 1.0 + time

    def termination_fn(state: float, time: float, step_index: int) -> bool:
        return step_index >= 2

    env = RLEnvironment(
        initial_state=0.0,
        dynamics=dynamics,
        reward_fn=reward_fn,
        termination_fn=termination_fn,
        dt=1.0,
    )
    agent = ConstantAgent(action=2.0)

    result = run_optimal_control_episode(env, agent, max_steps=10, start_time=0.0)

    assert len(result.steps) == 3
    assert len(agent.observed) == 3
    assert result.total_reward == pytest.approx(1.0 + 2.0 + 3.0)
    assert result.steps[0].state == pytest.approx(0.0)
    assert result.steps[-1].next_state == pytest.approx(6.0)
