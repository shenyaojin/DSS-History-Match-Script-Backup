"""Animate parameter history for the L1/lasso inversion."""

from animate_parameter_history_common import animate_parameter_history


if __name__ == "__main__":
    animate_parameter_history(
        csv_name="parameter_history_L1.csv",
        gif_name="parameter_history_L1.gif",
        label="L1",
    )
