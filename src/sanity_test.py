from src.Heston_Pricing_Integral_vectorized import price_heston
from src.VG_Pricing_Integral_vectorized import price_vg
import csv
import math


def read_r_data():
    with open("R_sanity_data.csv") as f:
        reader = csv.reader(f, delimiter=";")
        return [[str(item).replace(",", ".") for item in row] for row in reader if len(row) != 0]


def test():
    cases = read_r_data()
    for case in cases:
        model = str(case[0])
        data = case[1:-1]
        answer = case[-1]
        if model.lower() == "heston":
            func = price_heston
            pars = data[:5]
            args = data[5:]
        elif model.lower() == "vg":
            func = price_vg
            pars = data[:3]
            args = data[3:]
        else:
            raise Exception("Can't recognise model " + model)

        calculated = func(tuple(pars), tuple(args))
        if math.fabs(calculated - answer) > 1e-8:
            raise Exception("Sanity test failed with model: " + model + " and data:" +
                            ", ".join(list(map(lambda x: str(x), data))) +
                            "\nR answer: " + answer +
                            "\nPython answer: " + calculated)


if __name__ == "__main__":
    test()
