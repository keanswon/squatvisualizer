import cv2
import mediapipe as mp

class SquatVisualizer:
    """Handles all visualization and drawing operations"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
    
    def draw_clean_stick_figure(self, image, landmarks, min_visibility=0.3):
        """Draw clean stick figure focusing on reliable points"""
        if not landmarks:
            return image
        
        h, w, _ = image.shape
        
        # Get key landmarks
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Draw torso
        if all(lm.visibility > min_visibility for lm in [left_shoulder, right_shoulder, left_hip, right_hip]):
            shoulder_center_x = int((left_shoulder.x + right_shoulder.x) * w / 2)
            shoulder_center_y = int((left_shoulder.y + right_shoulder.y) * h / 2)
            hip_center_x = int((left_hip.x + right_hip.x) * w / 2)
            hip_center_y = int((left_hip.y + right_hip.y) * h / 2)
            
            # Draw spine
            cv2.line(image, (shoulder_center_x, shoulder_center_y), 
                    (hip_center_x, hip_center_y), (0, 255, 255), 3)
            
            # Draw shoulder and hip lines
            cv2.line(image, (int(left_shoulder.x * w), int(left_shoulder.y * h)),
                    (int(right_shoulder.x * w), int(right_shoulder.y * h)), (255, 0, 0), 2)
            cv2.line(image, (int(left_hip.x * w), int(left_hip.y * h)),
                    (int(right_hip.x * w), int(right_hip.y * h)), (255, 0, 0), 2)
        
        # Draw legs
        leg_connections = [
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE),
        ]
        
        for start_idx, end_idx in leg_connections:
            start_landmark = landmarks.landmark[start_idx]
            end_landmark = landmarks.landmark[end_idx]
            
            if start_landmark.visibility > min_visibility and end_landmark.visibility > min_visibility:
                start_coords = (int(start_landmark.x * w), int(start_landmark.y * h))
                end_coords = (int(end_landmark.x * w), int(end_landmark.y * h))
                cv2.line(image, start_coords, end_coords, (0, 255, 0), 3)
        
        # Draw joint points
        key_points = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]
        
        for point_idx in key_points:
            landmark = landmarks.landmark[point_idx]
            if landmark.visibility > min_visibility:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Color based on confidence
                if landmark.visibility > 0.7:
                    color = (0, 0, 255)  # Red - high confidence
                elif landmark.visibility > 0.5:
                    color = (0, 165, 255)  # Orange - medium confidence
                else:
                    color = (0, 255, 255)  # Yellow - low confidence
                
                cv2.circle(image, (x, y), 5, color, -1)
        
        return image
    
    def draw_form_feedback(self, image, analysis, depth_threshold=90):
        """Draw form analysis feedback on image"""
        h, w, _ = image.shape
        y_offset = 35
        
        # Detection quality
        quality = analysis.get('detection_quality', 'Unknown')
        visible = analysis.get('visible_landmarks', 0)
        color = (0, 255, 0) if quality == 'Good' else (0, 165, 255) if quality == 'Fair' else (0, 0, 255)
        cv2.putText(image, f'Detection: {quality} ({visible}/6)', (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 35
        
        # Knee angle and depth status
        if 'knee_angle' in analysis:
            angle = analysis['knee_angle']
            depth_status = analysis.get('depth_status', 'Unknown')
            
            # Color coding for depth status
            if depth_status == 'Depth reached':
                color = (0, 255, 0)  # Green
            elif depth_status == 'Not deep enough':
                color = (0, 0, 255)  # Red
            else:
                color = (255, 255, 255)  # White
            
            cv2.putText(image, f'Knee Angle: {int(round(angle))}°', (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 35
            
            cv2.putText(image, f'{depth_status.upper()}', (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            y_offset += 45
            
            # Individual knee angles
            if 'individual_knee_angles' in analysis:
                angles = analysis['individual_knee_angles']
                angle_text = f"L: {int(round(angles.get('left', 0)))}° R: {int(round(angles.get('right', 0)))}°"
                cv2.putText(image, angle_text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 40
        
        # Depth threshold reference
        cv2.putText(image, f'Target: <{depth_threshold}° for depth', 
                   (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return image
