"""Small timing helper used by the experiment scripts."""

import datetime as dt


class Timer:
    def __init__(self):
        self.start_dt = None
        self.elapsed = None

    def start(self):
        self.start_dt = dt.datetime.now()
        self.elapsed = None
        return self

    def stop(self):
        if self.start_dt is None:
            raise RuntimeError("Timer.stop() called before Timer.start()")
        end_dt = dt.datetime.now()
        self.elapsed = end_dt - self.start_dt
        print(f"Time taken: {self.elapsed}")
        return self.elapsed

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        self.stop()
        return False
