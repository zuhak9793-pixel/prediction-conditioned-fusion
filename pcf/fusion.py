"""OOF-fitted PCF and controlled readouts; all inputs and outputs are seconds."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def forward(kind, theta, nimf, b3):
    """Return prediction, parameter Jacobian, penalty and penalty gradient."""
    x = np.log1p(np.maximum(b3, 0))
    if kind == 'PCF':
        a, beta = theta
        slope = np.logaddexp(0, beta)
        weight = expit(a - slope * x)
        h = weight * (1 - weight) * (nimf - b3)
        prediction = b3 + weight * (nimf - b3)
        jacobian = np.column_stack([h, -h * x * expit(beta)])
        penalty = a * a + slope * slope
        gradient = np.array([2 * a, 2 * slope * expit(beta)])
    else:
        if kind == 'B3_nonlinear':
            jacobian = np.column_stack([np.ones(len(b3)), x])
            prediction = b3 + jacobian @ theta
        elif kind in ('constant_sigmoid', 'unconstrained_sigmoid'):
            weight = expit(theta[0] if kind == 'constant_sigmoid' else theta[0] + theta[1] * x)
            h = weight * (1 - weight) * (nimf - b3)
            jacobian = h[:, None] if kind == 'constant_sigmoid' else np.column_stack([h, h * x])
            prediction = b3 + weight * (nimf - b3)
        else:
            raise ValueError(f'Unknown readout: {kind}')
        penalty = theta @ theta
        gradient = 2 * theta
    return prediction, jacobian, penalty, gradient


def objective(kind, theta, nimf, b3, targets, config):
    prediction, jacobian, penalty, gradient = forward(kind, theta, nimf, b3)
    residual = prediction - targets
    delta = config['smooth_delta_s']
    root = np.hypot(residual, delta)
    regularization = config['regularization']
    return (float(np.mean(root - delta) + .5 * regularization * penalty),
            jacobian.T @ (residual / root) / len(targets) + .5 * regularization * gradient)


def fit(kind, nimf, b3, targets, config):
    initial = {'PCF': config['gate']['initial'], 'constant_sigmoid': [0.],
               'B3_nonlinear': [0., 0.],
               'unconstrained_sigmoid': [0., -float(np.logaddexp(0., -2.))]}[kind]
    optimizer = config['optimizer']
    result = minimize(lambda theta: objective(kind, theta, nimf, b3, targets, config),
                      np.array(initial), jac=True, method=optimizer['method'],
                      options={k: v for k, v in optimizer.items() if k != 'method'})
    value, gradient = objective(kind, result.x, nimf, b3, targets, config)
    record = dict(coefficients=result.x.tolist(), success=bool(result.success),
                  message=str(result.message), iterations=int(result.nit),
                  objective=value, max_abs_gradient=float(np.max(abs(gradient))))
    if not (result.success and np.isfinite(result.x).all()
            and record['max_abs_gradient'] <= config['accept_max_abs_gradient']):
        raise RuntimeError(f'Readout acceptance failed: {record}')
    return result.x, record


def predictions(nimf, b3, coefficients, development_ablation=False):
    out = dict(NIMF=nimf, B3=b3, fixed_half=.5 * nimf + .5 * b3)
    for kind, theta in coefficients.items():
        out[kind] = forward(kind, np.array(theta), nimf, b3)[0]
    if development_ablation:
        out.update(B3_clip=np.maximum(b3, 0),
                   clipped_fixed_half=.5 * nimf + .5 * np.maximum(b3, 0))
    return out
