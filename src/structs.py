class Info:
    def __init__(self, date: str, mat: str, spot: str, rate: str):
        self.date = date
        self.mat = int(mat)
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
        self.strikes = kwargs[self.__get_name({"k", "strikes", "strike", "x"}, kwargs.keys(), "strikes")]
        self.maturity = kwargs[self.__get_name({"m", "mat", "maturity", "tau", "t"}, kwargs.keys(), "maturity")]
        self.r = kwargs[self.__get_name({"r", "rate"}, kwargs.keys(), "r")]
        self.q = kwargs[self.__get_name({"q", "d"}, kwargs.keys(), "q")]
        self.is_call = kwargs[self.__get_name({"is_call", "call"}, kwargs.keys(), "is_call")]

    @staticmethod
    def __get_name(names: set, keys, param: str) -> str:
        found = names & keys
        if len(found) == 0:
            raise Exception(f"Cannot find {param} in {', '.join(keys)}")
        return found.pop()

    def as_tuple(self) -> tuple:
        return self.spot, self.strikes, self.maturity, self.r, self.q, self.is_call

    @staticmethod
    def from_tuple(t: tuple):
        args = EvalArgs()
        args.spot, args.strikes, args.maturity, args.r, args.q, args.is_call = t
        return args
