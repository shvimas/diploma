class Range:
    def __init__(self, num: int,
                 center: float = None, width: float = None,
                 p_min: float = None, p_max: float = None):
        if type(num) is not int or num < 1:
            raise ValueError("num must be >= 1")

        if p_min is None and p_max is None:
            self.min = center - width / 2
            self.max = center + width / 2
        elif center is None and width is None:
            self.min = p_min
            self.max = p_max
        else:
            raise ValueError("cannot crate Range with such parameters")
        self.num = num
        if self.num != 1:
            self.step = (self.max - self.min) / (num - 1)
        else:
            self.step = 0
            self.max = p_min

    def __iter__(self) -> float:
        for val in [self.min + i * self.step for i in range(0, self.num)]:
            yield val

    # noinspection PyMissingTypeHints
    def __getitem__(self, val):
        if type(val) is int:
            if val < 0:
                val += self.num
            return self.min + val * self.step

        if type(val) is slice:
            start = val.start
            if start is None:
                start = 0

            stop = val.stop
            if stop is None:
                stop = self.num
            if stop < 0:
                stop += self.num

            step = val.step
            if step is None:
                step = 1

            return [self.min + i * self.step for i in range(start, stop, step)]

        raise TypeError(f"Invalid argument type: {type(val)}")


class ParsRange:
    def __init__(self, model: str, center: tuple, widths: tuple, dots: int):
        self.center = center
        self.dots_per_dim = dots
        {
            "heston": lambda c, w: self.__init_heston(center=c, widths=w),
            "vg": lambda c, w: self.__init_vg(center=c, widths=w),
            "ls": lambda c, w: self.__init_ls(center=c, widths=w)
        }[model](center, widths)

    def __init_heston(self, center: tuple, widths: tuple):
        if len(center) != 5 or len(widths) != 5:
            raise ValueError("center and widths must be 5-dimensional")

        self.kappa_range = Range(center=center[0], width=widths[0], num=self.dots_per_dim)
        self.kappa = self.kappa_range.min

        self.theta_range = Range(center=center[1], width=widths[1], num=self.dots_per_dim)
        self.theta = self.theta_range.min

        self.sigma_range = Range(center=center[2], width=widths[2], num=self.dots_per_dim)
        self.sigma = self.sigma_range.min

        self.rho_range = Range(center=center[3], width=widths[3], num=self.dots_per_dim)
        self.rho = self.rho_range.min

        self.v0_range = Range(center=center[4], width=widths[4], num=self.dots_per_dim)
        self.v0 = self.v0_range.min

        self.is_bad_set = self.__is_bad4heston
        self.__iter_impl = self.__iter4heston

    def __init_vg(self, center: tuple, widths: tuple):
        if len(center) != 3 or len(widths) != 3:
            raise ValueError("center and widths must be 3-dimensional")

        self.nu_range = Range(center=center[0], width=widths[0], num=self.dots_per_dim)
        self.nu = self.nu_range.min

        self.theta_range = Range(center=center[1], width=widths[1], num=self.dots_per_dim)
        self.theta = self.theta_range.min

        self.sigma_range = Range(center=center[2], width=widths[2], num=self.dots_per_dim)
        self.sigma = self.sigma_range.min

        self.is_bad_set = self.__is_bad4vg
        self.__iter_impl = self.__iter4vg

    def __init_ls(self, center: tuple, widths: tuple):
        if len(center) != 2 or len(widths) != 2:
            raise ValueError("center and widths must be 2-dimensional")

        self.alpha_range = Range(center=center[0], width=widths[0], num=self.dots_per_dim)
        self.alpha = self.alpha_range.min

        self.sigma_range = Range(center=center[1], width=widths[1], num=self.dots_per_dim)
        self.sigma = self.sigma_range.min

        self.is_bad_set = self.__is_bad4ls
        self.__iter_impl = self.__iter4ls

    def __is_bad4heston(self) -> bool:
        result = 2 * self.kappa * self.theta <= self.sigma ** 2
        result |= self.sigma < 0
        result |= (self.rho < 0) | (self.rho > 1)
        result |= self.v0 < 0
        return result

    def __is_bad4vg(self) -> bool:
        result = self.theta ** 2 + (2 * self.sigma ** 2) / self.nu < 0
        result |= self.sigma <= 0
        result |= self.nu < 0
        return result

    def __is_bad4ls(self) -> bool:
        result = self.sigma <= 0
        result |= self.alpha <= 1
        result |= self.alpha > 2
        return result

    def __iter4heston(self) -> tuple:
        for self.kappa in self.kappa_range:
            for self.theta in self.theta_range:
                for self.sigma in self.sigma_range:
                    for self.rho in self.rho_range:
                        for self.v0 in self.v0_range:
                            if self.is_bad_set():
                                continue
                            yield self.kappa, self.theta, self.sigma, self.rho, self.v0

    def __iter4vg(self) -> tuple:
        for self.nu in self.nu_range:
            for self.theta in self.theta_range:
                for self.sigma in self.sigma_range:
                    if self.is_bad_set():
                        continue
                    yield self.nu, self.theta, self.sigma

    def __iter4ls(self) -> tuple:
        for self.alpha in self.alpha_range:
            for self.sigma in self.sigma_range:
                if self.is_bad_set():
                    continue
                yield self.alpha, self.sigma

    def __iter_impl(self):
        raise Exception('cannot iterate')

    def __iter__(self):
        return self.__iter_impl()
