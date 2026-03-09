import json
import sys

def convert_nb_to_py(nb_path, py_path):
    with open(nb_path, "r") as f:
        nb = json.load(f)
    with open(py_path, "w") as f:
        for i, cell in enumerate(nb["cells"]):
            f.write(f"# CELL {i}\n")
            if cell["cell_type"] == "code":
                for line in cell.get("source", []):
                    f.write(line)
                f.write("\n\n")

if __name__ == "__main__":
    convert_nb_to_py("improved_lstm_battery_thermal.ipynb", "improved_lstm_battery_thermal.py")
