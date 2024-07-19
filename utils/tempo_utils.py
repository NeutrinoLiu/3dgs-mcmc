import torch

class SliWinManager:
    def __init__(self, win_size, max_frame):
        self.start = 0
        self.end = win_size
        self.max = max_frame
    def tick(self):
        self.start += 1
        self.end += 1
    def fetch_cams(self, fetcher):
        ret = []
        for f in range(self.start, self.end):
            ret += fetcher(f).copy()
        return ret

def deform(gs, frame):
    return gs