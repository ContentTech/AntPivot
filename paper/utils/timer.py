import time
import math


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


class Timer(object):
    def __init__(self):
        super().__init__()
        self.start_time = self.last_time = None
        self.reset()

    def reset(self):
        self.start_time = self.last_time = time.time()

    @property
    def elapsed_total_time(self):
        return time.time() - self.start_time

    @property
    def elapsed_interval(self):
        current_time = time.time()
        interval = current_time - self.last_time
        self.last_time = current_time
        return interval

    def show_progress(self, percentage):
        return 'Running Time: {} ({}%)'.format(time_since(self.start_time, percentage), percentage * 100)
