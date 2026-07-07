import time

class TrackerState:
    """Manages the state of tracked objects (ByteTrack seen_ids) for the UI."""
    def __init__(self):
        self.seen_ids = {}      # track_id -> last_seen_timestamp
        self.pulse_frames = {}  # track_id -> frames remaining for pulse
        
    def reset(self):
        """Reset the seen IDs (e.g. on Stop or new camera source)."""
        self.seen_ids.clear()
        self.pulse_frames.clear()
        
    def check_and_add(self, track_id: int) -> bool:
        """
        Check if a track_id is new or expired.
        If it's new/expired, adds it to seen_ids and returns True.
        If it's already seen within the time limit or invalid (-1), returns False.
        """
        if track_id == -1:
            return False
            
        now = time.time()
        
        # FIX 2: Limit seen_ids size to prevent memory leaks / blocking
        if len(self.seen_ids) > 50:
            self.seen_ids.clear()
            self.pulse_frames.clear()
            
        # FIX 1: Time based reset (30 seconds)
        if track_id in self.seen_ids:
            last_seen = self.seen_ids[track_id]
            if (now - last_seen) > 30.0:
                # Treat as new detection, reset timer and pulse
                self.seen_ids[track_id] = now
                self.pulse_frames[track_id] = 4
                return True
            else:
                # Same track_id within 30 seconds -> skip and update timer
                self.seen_ids[track_id] = now
                return False
                
        # Completely new track_id
        self.seen_ids[track_id] = now
        self.pulse_frames[track_id] = 4
        return True
        
    def get_pulse_color(self, track_id: int, default_color: tuple) -> tuple:
        """
        Returns white if the track is currently pulsing, else default_color.
        Decrements the pulse frame counter.
        """
        if track_id in self.pulse_frames:
            frames_left = self.pulse_frames[track_id]
            if frames_left > 0:
                self.pulse_frames[track_id] -= 1
                return (255, 255, 255) # White
            else:
                del self.pulse_frames[track_id]
        return default_color
