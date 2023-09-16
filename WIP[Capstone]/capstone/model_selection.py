import numpy as np

def overunder_error(y_true, y_pred, underpred_penalty=1.0, overpred_penalty=1.0, alpha=0.5):
    """
    Calculate the Over-Under Error for portfolio optimization.

    Parameters:
        y_true (array-like): True portfolio values.
        y_pred (array-like): Predicted portfolio values.
        underpred_penalty (float, optional): Penalty factor for underpredictions. Default is 1.0.
        overpred_penalty (float, optional): Penalty factor for overpredictions. Default is 1.0.
        alpha (float, optional): Exponent for residual calculation. Default is 0.5.

    Returns:
        float: Mean Over-Under Error.

    The Over-Under Error is a custom loss function written for portfolio optimization. It calculates a loss based on the 
    differences between true and predicted portfolio values. The function allows for penalties on overpredictions and 
    underpredictions using the 'overpred_penalty' and 'underpred_penalty' parameters, respectively. The 'alpha' parameter
    determines the exponent for the residual calculation.

    If the residual (y_true - y_pred) is negative, it represents an underprediction, and the underprediction penalty is
    applied. If the residual is positive, it represents an overprediction, and the overprediction penalty is applied.

    The function returns the mean Over-Under Error, which can be used as an optimization objective in portfolio management.
    """
    # Calculate the residual between true and predicted values
    residual = y_true - y_pred
    
    # Define the loss using numpy's where function
    loss = np.where(
        residual < 0,  # If residual is negative (underprediction)
        underpred_penalty * np.abs(residual)**alpha,  # Apply underprediction penalty
        overpred_penalty * residual**alpha  # Apply overprediction penalty
    )
    
    # Calculate the mean of the loss
    return np.mean(loss)