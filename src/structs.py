

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
