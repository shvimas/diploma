class EvalArgs:
    def __init__(self, **kwargs):
        self.spot = kwargs[self.__get_name({"s", "spot", "price"}, kwargs.keys, "spot")]
        self.strikes = kwargs[self.__get_name({"k", "strikes", "strike", "x"}, kwargs.keys, "strikes")]
        self.maturity = kwargs[self.__get_name({"m", "mat", "maturity", "tau", "t"}, kwargs.keys, "maturity")]
        self.r = kwargs[self.__get_name({"r", "rate"}, kwargs.keys, "r")]
        self.q = kwargs[self.__get_name({"q", "d"}, kwargs.keys, "q")]
        self.is_call = kwargs[self.__get_name({"is_call", "call"}, kwargs.keys, "is_call")]

    @staticmethod
    def __get_name(names: set, keys, param: str) -> str:
        found = names & keys
        if len(found) == 0:
            raise Exception("Cannot find " + str(param) + " in " + keys)
        return found.pop()
