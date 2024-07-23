class SliWinManager:
    def __init__(self, win_size, max_frame):
        self.frame_start = 0
        self.frame_end = win_size
        self.max_frame = max_frame
    def tick(self):
        self.frame_start += 1
        self.frame_end += 1
    def fetch_cams(self, fetcher):
        ret = []
        for f in range(self.frame_start, self.frame_end):
            ret += fetcher(f).copy()
        return ret
    def frames(self):
        return range(self.frame_start, self.frame_end)

def deform(gs, frame):
    return gs