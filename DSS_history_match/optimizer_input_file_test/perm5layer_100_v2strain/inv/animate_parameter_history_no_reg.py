"""Animate parameter history for the no-regularization inversion."""

from animate_parameter_history_common import animate_parameter_history


if __name__ == "__main__":
    animate_parameter_history(
        csv_name="parameter_history_no_reg_fracture_init.csv",
        gif_name="parameter_history_no_reg_fracture_init.gif",
        label="No Reg",
    )
