import torch
import torch.jit

from .lif import LIFState, LIFParameters, lif_step

from .correlation_sensor import (
    CorrelationSensorState,
    CorrelationSensorParameters,
    correlation_sensor_step,
)

from typing import NamedTuple, Tuple


class LIFCorrelationState(NamedTuple):
    lif_state: LIFState
    input_correlation_state: CorrelationSensorState
    recurrent_correlation_state: CorrelationSensorState


class LIFCorrelationParameters(NamedTuple):
    lif_parameters: LIFParameters = LIFParameters()
    input_correlation_parameters: CorrelationSensorParameters = CorrelationSensorParameters()
    recurrent_correlation_parameters: CorrelationSensorParameters = CorrelationSensorParameters()


def lif_correlation_step(
    input: torch.Tensor,
    s: LIFCorrelationState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LIFCorrelationParameters = LIFCorrelationParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFCorrelationState]:
    z_new, s_new = lif_step(
        input, s.lif_state, input_weights, recurrent_weights, p.lif_parameters, dt
    )

    input_correlation_state_new = correlation_sensor_step(
        z_pre=input,
        z_post=z_new,
        s=s.input_correlation_state,
        p=p.input_correlation_parameters,
        dt=dt,
    )

    recurrent_correlation_state_new = correlation_sensor_step(
        z_pre=s.lif_state.z,
        z_post=z_new,
        s=s.recurrent_correlation_state,
        p=p.recurrent_correlation_parameters,
        dt=dt,
    )
    return (
        z_new,
        LIFCorrelationState(
            lif_state=s_new,
            input_correlation_state=input_correlation_state_new,
            recurrent_correlation_state=recurrent_correlation_state_new,
        ),
    )
