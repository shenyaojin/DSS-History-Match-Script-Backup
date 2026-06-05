"""Animate parameter history for the TV inversion."""

from animate_parameter_history_common import animate_parameter_history


if __name__ == "__main__":
    animate_parameter_history(
        csv_name="parameter_history_TV.csv",
        gif_name="parameter_history_TV.gif",
        label="TV",
    )
