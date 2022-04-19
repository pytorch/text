import time


class Timer:
    def __init__(self, text="") -> None:
        self._text = text
        self._start = None

    def start(self):
        if self._start is not None:
            raise Exception("Timer is already running. Call .stop() to stop it")

        self._start = time.perf_counter()

    def stop(self):
        if self._start is None:
            raise Exception("Timer is not running. Call .start() to start the timer.")

        elapsed = time.perf_counter() - self._start

        print(self._text + " ... Total running time: {}".format(elapsed))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info):
        self.stop()
