"""Reinforcement learning integration for optimal control of physical dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, Tuple, TypeVar

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")

DynamicsFn = Callable[[StateT, ActionT, float], StateT]
RewardFn = Callable[[StateT, ActionT, StateT, float], float]
TerminationFn = Callable[[StateT, float, int], bool]


@dataclass(frozen=True)
class EpisodeStep(Generic[StateT, ActionT]):
    """Single step transition in an optimal control rollout."""

    state: StateT
    action: ActionT
    next_state: StateT
    reward: float
    time: float


@dataclass(frozen=True)
class EpisodeResult(Generic[StateT, ActionT]):
    """Summary of an optimal control episode."""

    total_reward: float
    steps: List[EpisodeStep[StateT, ActionT]]


class RLEnvironment(Generic[StateT, ActionT]):
    """Minimal environment wrapper for physics dynamics and rewards."""

    def __init__(
        self,
        initial_state: StateT,
        dynamics: DynamicsFn[StateT, ActionT],
        reward_fn: RewardFn[StateT, ActionT],
        termination_fn: TerminationFn[StateT],
        dt: float = 0.01,
    ) -> None:
        self.initial_state = initial_state
        self.state = initial_state
        self.dynamics = dynamics
        self.reward_fn = reward_fn
        self.termination_fn = termination_fn
        self.dt = dt

    def reset(self, state: Optional[StateT] = None) -> StateT:
        """Reset the environment to the initial or provided state."""
        self.state = self.initial_state if state is None else state
        return self.state

    def step(self, action: ActionT, time: float, step_index: int) -> Tuple[StateT, float, bool]:
        """Advance dynamics by one step and return next state, reward, and done."""
        next_state = self.dynamics(self.state, action, self.dt)
        reward = self.reward_fn(self.state, action, next_state, time)
        done = self.termination_fn(next_state, time, step_index)
        self.state = next_state
        return next_state, reward, done


class OptimalControlAgent(Generic[StateT, ActionT]):
    """Base class for RL agents used in optimal control loops."""

    def act(self, state: StateT, time: float) -> ActionT:
        """Return an action given state and time."""
        raise NotImplementedError

    def observe(self, transition: EpisodeStep[StateT, ActionT]) -> None:
        """Receive a transition for learning updates."""
        return None


def run_optimal_control_episode(
    env: RLEnvironment[StateT, ActionT],
    agent: OptimalControlAgent[StateT, ActionT],
    max_steps: int = 1000,
    start_time: float = 0.0,
) -> EpisodeResult[StateT, ActionT]:
    """Roll out a single optimal control episode with an RL agent."""
    steps: List[EpisodeStep[StateT, ActionT]] = []
    total_reward = 0.0
    time = start_time
    env.reset()

    for step_index in range(max_steps):
        state = env.state
        action = agent.act(state, time)
        next_state, reward, done = env.step(action, time, step_index)
        transition = EpisodeStep(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            time=time,
        )
        agent.observe(transition)
        steps.append(transition)
        total_reward += reward
        time += env.dt
        if done:
            break

    return EpisodeResult(total_reward=total_reward, steps=steps)
