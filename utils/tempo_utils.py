from scene import DynamicScene
import random
class SliWinManager:
    def __init__(self, win_size, max_frame):
        self.frame_start = 0
        self.frame_end = win_size
        self.max_frame = max_frame
        self._sampled_frames = None

    def state_dump(self):
        return {
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "max_frame": self.max_frame,
            "_sampled_frames": self._sampled_frames
        }
    def state_load(self, state_dict):
        self.frame_start = state_dict["frame_start"]
        self.frame_end = state_dict["frame_end"]
        self.max_frame = state_dict["max_frame"]
        self._sampled_frames = state_dict["_sampled_frames"]
    def __str__(self):
        return f"window[{self.frame_start}:{self.frame_end}]"
    def tick(self):
        self.frame_start += 1
        self.frame_end += 1
    def fetch_cams(self, fetcher):
        return fetcher(self.sampled_frames()).copy()
    def sampled_frames(self, resample=False):
        if resample or (self._sampled_frames is None):
            self._sampled_frames = self.all_frames()
            if len(self._sampled_frames) > DynamicScene.MAX_FRAME_IN_MEMORY:
                self._sampled_frames = sorted(random.sample(self._sampled_frames, DynamicScene.MAX_FRAME_IN_MEMORY))
                print(f"Warning: too many frames in window, resample {DynamicScene.MAX_FRAME_IN_MEMORY} from {self}")
                print(f"Sampled frames: {self._sampled_frames}")
        return self._sampled_frames
    def all_frames(self):
        return range(self.frame_start, min(self.frame_end, self.max_frame))

def deform(gs, frame):
    return gs