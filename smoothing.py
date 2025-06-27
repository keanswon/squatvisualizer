import copy

class LandmarkSmoother:
    """Handles temporal smoothing of landmarks and angles"""
    
    def __init__(self, max_history=3, knee_angle_history_size=8):
        self.landmark_history = []
        self.max_history = max_history
        self.knee_angle_history = []
        self.knee_angle_history_size = knee_angle_history_size
        self.smoothed_knee_angle = None
        self.depth_history = []
        self.depth_history_size = 5
        self.current_depth_status = "Unknown"
        # Default threshold values - will be updated by set_thresholds()
        self.DEPTH_THRESHOLD = 90
        self.SHALLOW_THRESHOLD = 120

    def smooth_landmarks(self, landmarks):
        """Apply gentle temporal smoothing to reduce jitter"""
        if not landmarks:
            return None
        
        landmarks_copy = copy.deepcopy(landmarks)
            
        # Add current landmarks to history
        self.landmark_history.append(landmarks_copy)
        if len(self.landmark_history) > self.max_history:
            self.landmark_history.pop(0)
        
        # If we don't have enough history, return current
        if len(self.landmark_history) < 2:
            return landmarks
        
        # Light smoothing - weight current frame more heavily
        if len(self.landmark_history) == 2:
            # 70% current, 30% previous
            prev_landmarks = self.landmark_history[0]
            curr_landmarks = self.landmark_history[1]
            
            for i in range(len(landmarks.landmark)):
                curr_landmarks.landmark[i].x = (0.7 * curr_landmarks.landmark[i].x + 
                                               0.3 * prev_landmarks.landmark[i].x)
                curr_landmarks.landmark[i].y = (0.7 * curr_landmarks.landmark[i].y + 
                                               0.3 * prev_landmarks.landmark[i].y)
        
        return landmarks
    
    def smooth_knee_angle(self, new_angle):
        """Apply more aggressive smoothing to knee angle to reduce jitter"""
        if new_angle is None:
            return self.smoothed_knee_angle
        
        # Add current angle to history
        self.knee_angle_history.append(new_angle)
        if len(self.knee_angle_history) > self.knee_angle_history_size:
            self.knee_angle_history.pop(0)
        
        # Use exponential moving average for smoother results
        if self.smoothed_knee_angle is None:
            self.smoothed_knee_angle = new_angle
        else:
            # More aggressive smoothing - only 20% new data, 80% previous
            alpha = 0.2  # Lower = smoother but less responsive
            self.smoothed_knee_angle = alpha * new_angle + (1 - alpha) * self.smoothed_knee_angle
        
        return self.smoothed_knee_angle

    def smooth_depth_status(self, new_knee_angle):
        """Smooth depth status transitions to reduce text jumping"""
        if new_knee_angle is None:
            return self.current_depth_status
        
        # Add current angle to history
        self.depth_history.append(new_knee_angle)
        if len(self.depth_history) > self.depth_history_size:
            self.depth_history.pop(0)
        
        # Need at least 3 frames for smoothing
        if len(self.depth_history) < 3:
            return self.current_depth_status
        
        # Calculate average angle over recent frames
        avg_angle = sum(self.depth_history) / len(self.depth_history)
        
        # Determine status based on smoothed angle
        if avg_angle > self.DEPTH_THRESHOLD:
            new_status = 'Not deep enough'
        else:
            new_status = 'Depth reached'
        
        # Only change status if we have strong evidence (at least 3 consecutive frames suggesting change)
        recent_angles = self.depth_history[-3:]
        consistent_direction = True
        
        if self.current_depth_status != new_status:
            # Check if recent trend supports the change
            if new_status == 'Depth reached':
                consistent_direction = all(angle <= self.DEPTH_THRESHOLD + 5 for angle in recent_angles)
            else:
                consistent_direction = all(angle >= self.SHALLOW_THRESHOLD - 5 for angle in recent_angles)
            
            if consistent_direction:
                self.current_depth_status = new_status
        
        return self.current_depth_status
    
    def set_thresholds(self, depth_threshold, shallow_threshold):
        """Set depth thresholds from analyzer"""
        self.DEPTH_THRESHOLD = depth_threshold
        self.SHALLOW_THRESHOLD = shallow_threshold