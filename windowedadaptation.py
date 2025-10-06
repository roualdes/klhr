class WindowedAdaptation():
    def __init__(self, warmup, windowsize = 25, windowscale = 2):
        self._windowsize = windowsize
        self._windowscale = windowscale
        self._warmup = warmup
        self._closewindow = self._windowsize
        self._idx = 0
        self._closures = []
        self._num_windows = 0
        self._calculate_windows()

    def _calculate_windows(self):
        if self._warmup > self._windowsize:
            for w in range(self._warmup + 1):
                if w == self._closewindow:
                    self._closures.append(w)
                    self._calculate_next_window()
            self._num_windows = len(self._closures)

    def _calculate_next_window(self):
        self._windowsize *= self._windowscale
        nextclosewindow = self._closewindow + self._windowsize
        if self._closewindow + self._windowscale * self._windowsize >= self._warmup:
            self._closewindow = self._warmup
        else:
            self._closewindow = nextclosewindow

    def window_closed(self, m):
        if self._warmup < self._windowsize:
            return False
        closed = m == self._closures[self._idx]
        if closed and self._idx < self._num_windows - 1:
            self._idx += 1
        return closed

    def reset(self):
        self._idx = 0
        self._closures.clear()
        self._num_windows = 0
        self._calculate_windows()

if __name__ == "__main__":
    warmup = 15_000
    iterations = 30_000
    wa = WindowedAdaptation(warmup, windowsize = 50, windowscale = 2)
    closures = wa._closures
    for m in range(iterations):
        if wa.window_closed(m):
            print(m)

    wa.reset()
    print(wa._closures == closures)
