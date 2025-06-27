import numpy as np
import mediapipe as mp

class SquatAnalyzer:
    """Core squat form analysis logic"""
    
    def __init__(self, depth_threshold=90, shallow_threshold=120):
        self.DEPTH_THRESHOLD = depth_threshold
        self.SHALLOW_THRESHOLD = shallow_threshold
        self.mp_pose = mp.solutions.pose  # For landmark constants
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points (p2 is vertex)"""
        try:
            a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)
        except:
            return 0
    
    def analyze_squat_form(self, landmarks, min_visibility=0.3):
        """Analyze squat form focusing on knee angles"""
        if not landmarks:
            return {}
        
        # Get key landmarks
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        analysis = {}
        
        # Check landmark visibility
        required_landmarks = [left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]
        visible_count = sum(1 for lm in required_landmarks if lm.visibility > min_visibility)
        
        analysis['visible_landmarks'] = visible_count
        analysis['detection_quality'] = 'Good' if visible_count >= 5 else 'Poor' if visible_count < 3 else 'Fair'
        
        if visible_count < 4:
            analysis['form_issues'] = ['Insufficient landmark detection']
            return analysis
        
        # Calculate knee angles
        knee_angles = []
        individual_angles = {}
        
        # Left knee angle
        if all(lm.visibility > min_visibility for lm in [left_hip, left_knee, left_ankle]):
            left_knee_angle = self.calculate_angle(
                (left_hip.x, left_hip.y),
                (left_knee.x, left_knee.y),
                (left_ankle.x, left_ankle.y)
            )
            if left_knee_angle > 0:
                knee_angles.append(left_knee_angle)
                individual_angles['left'] = left_knee_angle
        
        # Right knee angle
        if all(lm.visibility > min_visibility for lm in [right_hip, right_knee, right_ankle]):
            right_knee_angle = self.calculate_angle(
                (right_hip.x, right_hip.y),
                (right_knee.x, right_knee.y),
                (right_ankle.x, right_ankle.y)
            )
            if right_knee_angle > 0:
                knee_angles.append(right_knee_angle)
                individual_angles['right'] = right_knee_angle
        
        # Store knee angle information
        if knee_angles:
            analysis['knee_angle'] = sum(knee_angles) / len(knee_angles)
            analysis['individual_knee_angles'] = individual_angles
        
        # Form issues
        analysis['form_issues'] = []
        
        # Check for knee angle imbalance
        if len(individual_angles) == 2:
            angle_diff = abs(individual_angles['left'] - individual_angles['right'])
            if angle_diff > 15:
                analysis['form_issues'].append(f'Uneven knee angles (diff: {angle_diff:.1f}°)')
        
        return analysis

