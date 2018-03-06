from typing import List


class Info:
    def __init__(self, date: str, mat: str, spot: str, rate: str):
        self.date = date
        self.mat = int(mat) / 365
        self.spot = float(spot)
        self.rate = float(rate)


class Data:
    def __init__(self, scall, sput, pcall, pput):
        self.prices = {True: pcall, False: pput}
        self.strikes = {True: scall, False: sput}


class EvalArgs:
    def __init__(self, args: dict = None, **kwargs):
        if args is not None:
            kwargs = args
        if len(kwargs) == 0:
            return
        self.spot = kwargs[self.__get_name({"s", "spot", "price"}, kwargs.keys(), "spot")]
        self.maturity = kwargs[self.__get_name({"m", "mat", "maturity", "tau", "t"}, kwargs.keys(), "maturity")]
        self.r = kwargs[self.__get_name({"r", "rate"}, kwargs.keys(), "r")]
        self.q = kwargs[self.__get_name({"q", "d"}, kwargs.keys(), "q")]
        try:
            self.is_call = kwargs[self.__get_name({"is_call", "call"}, kwargs.keys(), "is_call")]
            if self.is_call is None:
                raise ValueError
        except ValueError:
            self.is_call = None  # have both calls and puts
            self.strikes_call = kwargs[self.__get_name(
                {"k_call", "strikes_call", "strike_call", "x_call"}, kwargs.keys(), "strikes_call")]
            self.strikes_put = kwargs[self.__get_name(
                {"k_put", "strikes_put", "strike_put", "x_put"}, kwargs.keys(), "strikes_put")]
            return
        if self.is_call:
            self.strikes_call = kwargs[self.__get_name(
                {"k_call", "strikes_call", "strike_call", "x_call"}, kwargs.keys(), "strikes_call")]
        else:
            self.strikes_put = kwargs[self.__get_name(
                {"k_put", "strikes_put", "strike_put", "x_put"}, kwargs.keys(), "strikes_put")]

    @staticmethod
    def __get_name(names: set, keys, param: str) -> str:
        found = names & keys
        if len(found) == 0:
            raise ValueError(f"Cannot find {param} in {', '.join(keys)}")
        return found.pop()

    def as_tuple(self) -> tuple:
        if self.is_call is None:
            return self.spot, self.strikes_call, self.strikes_put, self.maturity, self.r, self.q, self.is_call
        if self.is_call:
            return self.spot, self.strikes_call, self.maturity, self.r, self.q, self.is_call
        return self.spot, self.strikes_put, self.maturity, self.r, self.q, self.is_call

    @staticmethod
    def from_tuple(t: tuple):
        args = EvalArgs()
        if t[-1] is None:  # t[-1] == args.is_call
            args.spot, args.strikes_call, args.strikes_put, args.maturity, args.r, args.q, args.is_call = t
        elif t[-1]:
            args.spot, args.strikes_call, args.maturity, args.r, args.q, args.is_call = t
        else:
            args.spot, args.strikes_put, args.maturity, args.r, args.q, args.is_call = t
        return args

    @staticmethod
    def from_structure(data: Data, info: List[Info], rate: float, day: int):
        args = EvalArgs()
        args.spot = info[day].spot
        args.strikes_call = data.strikes[True][day]
        args.strikes_put = data.strikes[False][day]
        args.maturity = info[day].mat
        args.r = rate
        args.q = rate
        args.is_call = None
        return args

    def get_strikes(self):
        if self.is_call is not None:
            if self.is_call:
                return self.strikes_call
            else:
                return self.strikes_put
        raise ValueError('have both call and put strikes')
