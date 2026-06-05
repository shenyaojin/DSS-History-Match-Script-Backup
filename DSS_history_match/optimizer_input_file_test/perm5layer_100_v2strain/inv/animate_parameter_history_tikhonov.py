"""Animate parameter history for the L2/Tikhonov inversion."""

from animate_parameter_history_common import animate_parameter_history


if __name__ == "__main__":
    animate_parameter_history(
        csv_name="parameter_history.csv",
        gif_name="parameter_history_tikhonov.gif",
        label="L2 Tikhonov",
    )
