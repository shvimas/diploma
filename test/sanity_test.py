from heston_pricing import price_heston
from vg_pricing import price_vg
from ls_pricing import price_ls
import csv
from typing import List


def read_r_data():
    with open("R_sanity_data.csv") as f:
        reader = csv.reader(f, delimiter=";")
        return [[str(item).replace(",", ".") for item in row] for row in reader if len(row) != 0]


def test():

    # noinspection PyShadowingNames
    def prepare_args(a: List[str]) -> tuple:
        spot = float(a[0])
        strike = float(a[1])
        tau = float(a[2])
        r = float(a[3])
        q = float(a[4])
        is_call = True if a[5] == 'TRUE' else False
        return spot, strike, tau, r, q, is_call

    def params2tuple(params: tuple) -> tuple:
        return params + tuple(None for _ in range(5 - len(params)))

    cases = read_r_data()

    names = ['model', 'p1', 'p2', 'p3', 'p4', 'p5',
             'spot', 'strike', 't', 'r', 'q', 'is call',
             'R answer', 'Python answer', 'diff', 'rel diff', 'is correct']

    with open('sanity_log.txt', 'w') as log, open('sanity.csv', 'w') as out:
        out.write('; '.join(names) + "\n")

        for i, case in enumerate(cases):
            correct = True
            model = str(case[0])
            data = case[1:-1]
            answer = float(case[-1])
            if model.lower() == "heston":
                func = price_heston
                pars = tuple(map(lambda x: float(x), data[:5]))
                args = prepare_args(data[5:])
            elif model.lower() == "vg":
                func = price_vg
                pars = tuple(map(lambda x: float(x), data[:3]))
                args = prepare_args(data[3:])
            elif model.lower() == "ls":
                func = price_ls
                pars = tuple(map(lambda x: float(x), data[:2]))
                args = prepare_args(data[2:])
            else:
                raise Exception(f"Can't recognise model {model}")

            calculated = float(func(pars, args))
            diff = abs(answer - calculated)
            if diff > 1e-2 * answer and diff > 1e-3:
                correct = False
                log.write(f"Sanity test failed with case: {', '.join(case)}\n"
                          f"\tR answer: {answer}\n"
                          f"\tPython answer: {calculated}\n"
                          f"\tDiff: {diff}\n\n")
                log.flush()

            p1, p2, p3, p4, p5 = params2tuple(params=pars)
            spot, strike, t, r, q, is_call = args
            row = f'{model};{p1};{p2};{p3};{p4};{p5};{spot};{strike};{t};{r};{q};' \
                  f'{is_call};{answer};{calculated};{diff};' \
                  f'{diff / answer if answer != 0 else "Inf"};{"+" if correct else "-"}\n'
            out.write(row)
            out.flush()

            if i % 1000 == 0:
                print(f"{i / len(cases):.{3}}")


if __name__ == "__main__":
    test()
