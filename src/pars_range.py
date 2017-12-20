class Range:
    def __init__(self, center: float, width: float, num: int):
        self.min = center - width / 2
        self.max = center + width / 2
        self.num = num
        self.step = (self.max - self.min) / num

    def __iter__(self) -> float:
        for val in [self.min + i * self.step for i in range(0, self.num)]:
            yield val


class ParsRange:
    def __init__(self, model: str, center: tuple, dots: int):
        self.center = center
        self.dots = dots
        {
            "heston": lambda c: self.__init_heston(c),
            "vg": lambda c: self.__init_vg(c)
        }[model](center)

    def __init_heston(self, center: tuple):
        self.dots_per_dim = int(self.dots ** .2)

        self.kappa_range = Range(center=center[0], width=1, num=self.dots_per_dim)
        self.kappa = self.kappa_range.min

        self.theta_range = Range(center=center[1], width=1, num=self.dots_per_dim)
        self.theta = self.theta_range.min

        self.sigma_range = Range(center=center[2], width=.2, num=self.dots_per_dim)
        self.sigma = self.sigma_range.min

        self.rho_range = Range(center=center[3], width=.3, num=self.dots_per_dim)
        self.rho = self.rho_range.min

        self.v0_range = Range(center=center[4], width=.2, num=self.dots_per_dim)
        self.v0 = self.v0_range.min

        self.is_bad_set = self.__is_bad4heston
        self.__gen_impl = self.__gen4heston

    def __init_vg(self, center: tuple):
        self.dots_per_dim = int(self.dots ** (1 / 3))

        self.nu_range = Range(center[0], .5, self.dots_per_dim)
        self.nu = self.nu_range.min

        self.theta_range = Range(center[1], 1, self.dots_per_dim)
        self.theta = self.theta_range.min

        self.sigma_range = Range(center[2], .3, self.dots_per_dim)
        self.sigma = self.sigma_range.min

        self.is_bad_set = self.__is_bad4vg
        self.__gen_impl = self.__gen4vg

    def __is_bad4heston(self) -> bool:
        result = 2 * self.kappa * self.theta <= self.sigma ** 2
        result |= self.sigma < 0
        result |= (self.rho < 0) | (self.rho > 1)
        result |= self.v0 < 0
        return result

    def __is_bad4vg(self) -> bool:
        result = self.theta ** 2 + (2 * self.sigma ** 2) / self.nu < 0
        result |= self.sigma < 0
        result |= self.nu < 0
        return result

    def __gen4heston(self) -> tuple:
        for self.kappa in self.kappa_range:
            for self.theta in self.theta_range:
                for self.sigma in self.sigma_range:
                    for self.rho in self.rho_range:
                        for self.v0 in self.v0_range:
                            if self.is_bad_set():
                                continue
                            yield self.kappa, self.theta, self.sigma, self.rho, self.v0

    def __gen4vg(self) -> tuple:
        for self.nu in self.nu_range:
            for self.theta in self.theta_range:
                for self.sigma in self.sigma_range:
                    if self.is_bad_set():
                        continue
                    yield self.nu, self.theta, self.sigma

    def generate(self):
        return self.__gen_impl()
