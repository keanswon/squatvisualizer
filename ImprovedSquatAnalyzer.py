import cv2
import mediapipe as mp
import numpy as np

# Global pose detector instance for faster loading
_pose_detector = None

def get_pose_detector():
    global _pose_detector
    if _pose_detector is None:
        mp_pose = mp.solutions.pose
        _pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Reduced complexity for better performance
            enable_segmentation=False,
            min_detection_confidence=0.5,  # Lower threshold for better detection
            min_tracking_confidence=0.65
        )
    return _pose_detector

class ImprovedSquatAnalyzer:
    def __init__(self):
        # Initialize MediaPipe pose detection with singleton pattern
        self.mp_pose = mp.solutions.pose
        self.pose = get_pose_detector()
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Store previous landmarks for smoothing
        self.prev_landmarks = None
        self.landmark_history = []
        self.max_history = 3  # Reduced history for less lag

        # Knee angle smoothing
        self.knee_angle_history = []
        self.knee_angle_history_size = 8  # Smooth over 8 frames for stability
        self.smoothed_knee_angle = None

        self.depth_history = []
        self.depth_history_size = 5  # Number of frames to consider for smoothing
        self.current_depth_status = "Unknown"
        
        # Frame counter for debugging
        self.frame_count = 0
        self.detection_count = 0
        
        # Squat depth thresholds (knee angles)
        self.DEPTH_THRESHOLD = 90  # Knee angle threshold for "depth reached"
        self.SHALLOW_THRESHOLD = 120  # Above this is considered shallow

    def smooth_landmarks(self, landmarks):
        """Apply gentle temporal smoothing to reduce jitter"""
        if not landmarks:
            return None
            
        # Add current landmarks to history
        self.landmark_history.append(landmarks)
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

    def filter_reliable_landmarks(self, landmarks, threshold=0.3):
        """Only use landmarks with reasonable confidence - lowered threshold"""
        reliable_indices = []
        for i, landmark in enumerate(landmarks.landmark):
            if landmark.visibility > threshold:
                reliable_indices.append(i)
        return reliable_indices
    
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

    def draw_clean_stick_figure(self, image, landmarks):
        """Draw a cleaner stick figure focusing on reliable points"""
        if not landmarks:
            return image
        
        h, w, _ = image.shape
        min_visibility = 0.3  # Much lower threshold
        
        # Get key landmarks
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Draw torso if shoulders and hips are visible
        if (left_shoulder.visibility > min_visibility and right_shoulder.visibility > min_visibility and
            left_hip.visibility > min_visibility and right_hip.visibility > min_visibility):
            
            # Calculate center points
            shoulder_center_x = int((left_shoulder.x + right_shoulder.x) * w / 2)
            shoulder_center_y = int((left_shoulder.y + right_shoulder.y) * h / 2)
            hip_center_x = int((left_hip.x + right_hip.x) * w / 2)
            hip_center_y = int((left_hip.y + right_hip.y) * h / 2)
            
            # Draw spine
            cv2.line(image, (shoulder_center_x, shoulder_center_y), 
                    (hip_center_x, hip_center_y), (0, 255, 255), 3)  # Yellow spine
            
            # Draw shoulder line
            cv2.line(image, (int(left_shoulder.x * w), int(left_shoulder.y * h)),
                    (int(right_shoulder.x * w), int(right_shoulder.y * h)), (255, 0, 0), 2)
            
            # Draw hip line
            cv2.line(image, (int(left_hip.x * w), int(left_hip.y * h)),
                    (int(right_hip.x * w), int(right_hip.y * h)), (255, 0, 0), 2)
        
        # Draw legs - prioritize these for squat analysis
        leg_connections = [
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE, "Left Thigh"),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE, "Left Shin"),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE, "Right Thigh"),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE, "Right Shin"),
        ]
        
        for start_idx, end_idx, name in leg_connections:
            start_landmark = landmarks.landmark[start_idx]
            end_landmark = landmarks.landmark[end_idx]
            
            if start_landmark.visibility > min_visibility and end_landmark.visibility > min_visibility:
                start_coords = (int(start_landmark.x * w), int(start_landmark.y * h))
                end_coords = (int(end_landmark.x * w), int(end_landmark.y * h))
                cv2.line(image, start_coords, end_coords, (0, 255, 0), 3)
        
        # Draw joint points with different colors based on confidence
        key_points = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, "L_Shoulder"),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, "R_Shoulder"),
            (self.mp_pose.PoseLandmark.LEFT_HIP, "L_Hip"),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, "R_Hip"),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, "L_Knee"),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, "R_Knee"),
            (self.mp_pose.PoseLandmark.LEFT_ANKLE, "L_Ankle"),
            (self.mp_pose.PoseLandmark.RIGHT_ANKLE, "R_Ankle"),
        ]
        
        for point_idx, name in key_points:
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
                
                # Add debug text for key leg points
                if 'Knee' in name or 'Hip' in name or 'Ankle' in name:
                    cv2.putText(image, f'{landmark.visibility:.2f}', 
                               (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return image

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points (p2 is the vertex)"""
        try:
            # Create vectors from the vertex point (p2) to the other two points
            a = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            b = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Calculate the angle using dot product
            cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)
        except:
            return 0

    def analyze_squat_form(self, landmarks):
        """Improved squat form analysis focusing on knee angles"""
        if not landmarks:
            return {}
        
        min_visibility = 0.3  # Lower threshold
        
        # Get key landmarks
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        analysis = {}
        
        # Check if we have enough landmarks for analysis
        required_landmarks = [left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]
        visible_count = sum(1 for lm in required_landmarks if lm.visibility > min_visibility)
        
        analysis['visible_landmarks'] = visible_count
        analysis['detection_quality'] = 'Good' if visible_count >= 5 else 'Poor' if visible_count < 3 else 'Fair'
        
        if visible_count < 4:
            analysis['form_issues'] = ['Insufficient landmark detection']
            return analysis
        
        # Calculate knee angles (this is the key metric for squat depth)
        knee_angles = []
        individual_angles = {}
        
        # Left knee angle
        if (left_hip.visibility > min_visibility and left_knee.visibility > min_visibility and 
            left_ankle.visibility > min_visibility):
            left_knee_angle = self.calculate_angle(
                (left_hip.x, left_hip.y),
                (left_knee.x, left_knee.y),
                (left_ankle.x, left_ankle.y)
            )
            if left_knee_angle > 0:
                knee_angles.append(left_knee_angle)
                individual_angles['left'] = left_knee_angle
        
        # Right knee angle
        if (right_hip.visibility > min_visibility and right_knee.visibility > min_visibility and 
            right_ankle.visibility > min_visibility):
            right_knee_angle = self.calculate_angle(
                (right_hip.x, right_hip.y),
                (right_knee.x, right_knee.y),
                (right_ankle.x, right_ankle.y)
            )
            if right_knee_angle > 0:
                knee_angles.append(right_knee_angle)
                individual_angles['right'] = right_knee_angle
        
        # Store knee angle information
        # Store knee angle information with smoothing
        if knee_angles:
            raw_angle = sum(knee_angles) / len(knee_angles)
            smoothed_angle = self.smooth_knee_angle(raw_angle)
            analysis['knee_angle'] = smoothed_angle
            analysis['raw_knee_angle'] = raw_angle  # Keep raw for debugging
            analysis['individual_knee_angles'] = individual_angles
        else:
            # No angle detected, use previous smoothed value
            analysis['knee_angle'] = self.smooth_knee_angle(None)
        
        # Depth assessment based on knee angles
        # Depth assessment based on knee angles with smoothing
        analysis['form_issues'] = []

        if 'knee_angle' in analysis:
            avg_knee_angle = analysis['knee_angle']
            
            # Get smoothed depth status
            analysis['depth_status'] = self.smooth_depth_status(avg_knee_angle)
            
            # Add form issues based on smoothed status
            if analysis['depth_status'] == 'Not deep enough':
                analysis['form_issues'].append('Not deep enough - squat deeper')

            # No issue added for 'Depth reached'
        else:
            # No knee angle available, use previous status
            analysis['depth_status'] = self.smooth_depth_status(None)
        
        # Check for knee angle imbalance
        if len(individual_angles) == 2:
            angle_diff = abs(individual_angles['left'] - individual_angles['right'])
            if angle_diff > 15:  # More than 15 degrees difference
                analysis['form_issues'].append(f'Uneven knee angles (diff: {angle_diff:.1f}°)')
        
        return analysis

    def draw_form_feedback(self, image, analysis):
        """Draw form analysis on the image with improved depth feedback"""
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
            
            # Large, prominent depth status
            cv2.putText(image, f'{depth_status.upper()}', (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            y_offset += 45
            
            # Show individual knee angles if available
            if 'individual_knee_angles' in analysis:
                angles = analysis['individual_knee_angles']
                angle_text = f"L: {int(round(angles.get('left', 0)))}° R: {int(round(angles.get('right', 0)))}°"
                cv2.putText(image, angle_text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # White and thicker
                y_offset += 40
        
        # ignore form issues for now
        # # Form issues
        # if analysis.get('form_issues'):
        #     cv2.putText(image, 'Issues:', (10, y_offset), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #     y_offset += 25
            
        #     for issue in analysis['form_issues'][:2]:  # Show max 2 issues
        #         cv2.putText(image, f'• {issue}', (15, y_offset), 
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        #         y_offset += 20
        # else:
        #     if analysis.get('detection_quality') == 'Good' and analysis.get('depth_status') == 'Depth reached':
        #         cv2.putText(image, 'Great form!', (10, y_offset), 
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Depth threshold reference
        cv2.putText(image, f'Target: <{self.DEPTH_THRESHOLD}° for depth', 
                   (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Debug info
        cv2.putText(image, f'Frame: {self.frame_count} | Detections: {self.detection_count}', 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return image
    
    def analyze_back_posture(self, landmarks):
        """
        Modified back analysis that focuses on spinal flexion (rounding) rather than forward lean.
        Forward lean is normal and expected in squats, especially low-bar squats.
        """
        if not landmarks:
            return {}
        
        min_visibility = 0.3
        back_analysis = {}
        
        # Get spine-related landmarks
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Try to get head/neck landmarks for better curvature analysis
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        
        # Check if we have enough landmarks for back analysis
        spine_landmarks = [left_shoulder, right_shoulder, left_hip, right_hip]
        visible_spine_count = sum(1 for lm in spine_landmarks if lm.visibility > min_visibility)
        
        back_analysis['spine_landmarks_visible'] = visible_spine_count
        
        if visible_spine_count < 3:
            back_analysis['back_analysis_available'] = False
            back_analysis['back_status'] = 'Insufficient data'
            return back_analysis
        
        back_analysis['back_analysis_available'] = True
        
        try:
            # Calculate spine alignment using shoulder and hip centers
            if (left_shoulder.visibility > min_visibility and right_shoulder.visibility > min_visibility and
                left_hip.visibility > min_visibility and right_hip.visibility > min_visibility):
                
                # Calculate center points
                shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
                shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
                hip_center_x = (left_hip.x + right_hip.x) / 2
                hip_center_y = (left_hip.y + right_hip.y) / 2
                
                # Store the basic spine vector for reference
                spine_vector_x = shoulder_center_x - hip_center_x
                spine_vector_y = shoulder_center_y - hip_center_y
                
                # Method 1: MAIN ANALYSIS - Spinal curvature using head position
                if nose.visibility > min_visibility:
                    # Three-point analysis: head → shoulders → hips
                    upper_point = (nose.x, nose.y)
                    mid_point = (shoulder_center_x, shoulder_center_y)
                    lower_point = (hip_center_x, hip_center_y)
                    
                    # Calculate actual spinal curvature (flexion)
                    curvature_deviation = self.calculate_spine_curvature(upper_point, mid_point, lower_point)
                    back_analysis['spine_curvature'] = curvature_deviation
                    
                    # More conservative thresholds - only flag obvious rounding
                    if curvature_deviation > 0.12:  # Higher threshold - only severe rounding
                        back_analysis['back_status'] = 'Significant back rounding'
                        back_analysis['back_issue'] = True
                    elif curvature_deviation > 0.08:  # Moderate threshold
                        back_analysis['back_status'] = 'Mild back rounding'
                        back_analysis['back_issue'] = True
                    else:
                        back_analysis['back_status'] = 'Spine neutral'
                        back_analysis['back_issue'] = False
                        
                    # Additional context: Calculate the expected vs actual head position
                    # In a neutral spine, head should be roughly in line with the spine angle
                    expected_head_x = shoulder_center_x + (shoulder_center_x - hip_center_x) * 0.3
                    actual_head_deviation = abs(nose.x - expected_head_x)
                    back_analysis['head_position_deviation'] = actual_head_deviation
                    
                    # Only flag if head is significantly forward AND we have curvature
                    if actual_head_deviation > 0.08 and curvature_deviation > 0.06:
                        if not back_analysis.get('back_issue', False):
                            back_analysis['back_status'] = 'Head forward posture'
                            back_analysis['back_issue'] = True
                
                else:
                    # Fallback method when head is not visible - shoulder position analysis
                    # Look for shoulder protraction (rounding forward of shoulders)
                    
                    # Calculate shoulder width and compare to hip width
                    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
                    hip_width = abs(left_hip.x - right_hip.x)
                    
                    # In rounded posture, shoulders often appear narrower due to protraction
                    width_ratio = shoulder_width / hip_width if hip_width > 0 else 1.0
                    back_analysis['shoulder_width_ratio'] = width_ratio
                    
                    # Conservative threshold - only flag obvious shoulder protraction
                    if width_ratio < 0.85:  # Shoulders significantly narrower than hips
                        back_analysis['back_status'] = 'Possible shoulder protraction'
                        back_analysis['back_issue'] = True
                    else:
                        back_analysis['back_status'] = 'Shoulder position OK'
                        back_analysis['back_issue'] = False
            
            # Overall back assessment - more conservative
            if 'back_issue' not in back_analysis:
                back_analysis['back_issue'] = False
                back_analysis['back_status'] = 'Spine neutral'
        
        except Exception as e:
            back_analysis['back_analysis_error'] = str(e)
            back_analysis['back_status'] = 'Analysis error'
        
        return back_analysis

    def calculate_spine_curvature(self, upper_point, mid_point, lower_point):
        """
        Calculate spinal curvature focusing on flexion rather than normal postural angles.
        This version is more specific to detecting actual rounding vs normal squat posture.
        """
        try:
            # Convert to numpy arrays
            p1 = np.array(upper_point)    # Head/nose
            p2 = np.array(mid_point)      # Shoulder center
            p3 = np.array(lower_point)    # Hip center
            
            # Calculate the direct line from head to hips
            head_to_hip_vector = p3 - p1
            head_to_hip_length = np.linalg.norm(head_to_hip_vector)
            
            if head_to_hip_length == 0:
                return 0
            
            # Vector from head to shoulders
            head_to_shoulder = p2 - p1
            
            # Project shoulder position onto the head-hip line
            projection_length = np.dot(head_to_shoulder, head_to_hip_vector) / head_to_hip_length
            projection_point = p1 + (projection_length / head_to_hip_length) * head_to_hip_vector
            
            # Distance from actual shoulder position to the straight head-hip line
            deviation = np.linalg.norm(p2 - projection_point)
            
            # Normalize by the head-hip distance
            normalized_deviation = deviation / head_to_hip_length
            
            # Additional check: determine if this is forward deviation (rounding)
            # vs backward deviation (over-extension)
            cross_product = np.cross(head_to_hip_vector[:2], head_to_shoulder[:2])
            
            # Only return positive values for forward deviation (actual rounding)
            if cross_product > 0:  # Shoulder is forward of the line (rounding)
                return normalized_deviation
            else:  # Shoulder is behind the line (extension/neutral)
                return normalized_deviation * 0.5  # Reduce the penalty for extension
        
        except:
            return 0

    def draw_back_analysis_feedback(self, image, back_analysis, y_start_offset=200):
        """
        Modified feedback display focusing on actual spinal issues rather than normal squat posture.
        """
        if not back_analysis.get('back_analysis_available', False):
            return image, y_start_offset
        
        h, w, _ = image.shape
        y_offset = y_start_offset
        
        # Back posture status
        back_status = back_analysis.get('back_status', 'Unknown')
        back_issue = back_analysis.get('back_issue', False)
        
        # Color coding - more nuanced
        if back_issue and 'Significant' in back_status:
            color = (0, 0, 255)  # Red for significant issues
            status_prefix = "⚠ "
        elif back_issue:
            color = (0, 165, 255)  # Orange for mild issues
            status_prefix = "⚠ "
        else:
            color = (0, 255, 0)  # Green for good posture
            status_prefix = "✓ "
        
        cv2.putText(image, f'{status_prefix}Spine: {back_status}', (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30
        
        # Show curvature metric if available (for debugging/fine-tuning)
        if 'spine_curvature' in back_analysis:
            curvature = back_analysis['spine_curvature']
            # Only show if there's some curvature detected
            if curvature > 0.03:
                cv2.putText(image, f'Curvature: {curvature:.3f}', (15, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 25
        
        # Show shoulder width ratio if head not visible
        if 'shoulder_width_ratio' in back_analysis:
            ratio = back_analysis['shoulder_width_ratio']
            cv2.putText(image, f'Shoulder ratio: {ratio:.2f}', (15, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
        
        # Only show asymmetry if it's severe
        if back_analysis.get('asymmetry_severe', False):
            cv2.putText(image, '• Severe shoulder asymmetry', (15, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += 25
        
        return image, y_offset

    def draw_spine_visualization(self, image, landmarks, back_analysis):
        """
        Enhanced spine visualization that shows normal squat posture vs rounding.
        Forward lean is shown as normal (yellow/green) while rounding is red.
        """
        if not landmarks or not back_analysis.get('back_analysis_available', False):
            return image
        
        h, w, _ = image.shape
        min_visibility = 0.3
        
        # Get landmarks
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        
        if (left_shoulder.visibility > min_visibility and right_shoulder.visibility > min_visibility and
            left_hip.visibility > min_visibility and right_hip.visibility > min_visibility):
            
            # Calculate centers
            shoulder_center_x = int((left_shoulder.x + right_shoulder.x) * w / 2)
            shoulder_center_y = int((left_shoulder.y + right_shoulder.y) * h / 2)
            hip_center_x = int((left_hip.x + right_hip.x) * w / 2)
            hip_center_y = int((left_hip.y + right_hip.y) * h / 2)
            
            # Color spine based on actual rounding, not forward lean
            back_issue = back_analysis.get('back_issue', False)
            spine_curvature = back_analysis.get('spine_curvature', 0)
            
            # Color logic: Green/yellow for normal forward lean, red only for actual rounding
            if back_issue and spine_curvature > 0.08:
                spine_color = (0, 0, 255)  # Red for actual rounding
                thickness = 4
            else:
                spine_color = (0, 255, 255)  # Yellow for normal posture (even with forward lean)
                thickness = 3
            
            # Draw spine line
            cv2.line(image, (shoulder_center_x, shoulder_center_y), 
                    (hip_center_x, hip_center_y), spine_color, thickness)
            
            # If head is visible and we're analyzing curvature
            if nose.visibility > min_visibility and 'spine_curvature' in back_analysis:
                nose_x = int(nose.x * w)
                nose_y = int(nose.y * h)
                
                # Draw head-to-shoulder line
                cv2.line(image, (nose_x, nose_y), (shoulder_center_x, shoulder_center_y), spine_color, 2)
                
                # Add annotation for significant curvature
                if spine_curvature > 0.08:
                    cv2.putText(image, f'Rounding: {spine_curvature:.3f}', 
                            (shoulder_center_x + 10, shoulder_center_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, spine_color, 1)
            
            # Add a small indicator for "normal forward lean"
            if not back_issue:
                cv2.putText(image, 'Normal lean', 
                        (shoulder_center_x + 10, shoulder_center_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return image
    
    def process_video(self, video_path, output_path=None):
        """Process video with improved analysis"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # NEW: Debug video properties
        print(f"=== VIDEO DEBUG INFO ===")
        print(f"Reported dimensions: {width}x{height}")
        print(f"Aspect ratio: {width/height:.2f}")
        print(f"Is vertical: {height > width}")
        
        # Check if there's rotation metadata
        rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        print(f"Rotation metadata: {rotation}")
        
        # Read first frame to check actual dimensions
        ret, first_frame = cap.read()
        if ret:
            actual_height, actual_width = first_frame.shape[:2]
            print(f"Actual frame dimensions: {actual_width}x{actual_height}")
            print(f"Dimensions match: {width == actual_width and height == actual_height}")
            
            # Reset video to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print(f"========================")
        
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        print(f"Squat depth thresholds: Depth at <{self.DEPTH_THRESHOLD}°, Shallow at >{self.SHALLOW_THRESHOLD}°")
        
        # If you want to force the output to maintain the input orientation:
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Use the ACTUAL frame dimensions, not the reported ones
            if 'actual_width' in locals() and 'actual_height' in locals():
                out = cv2.VideoWriter(output_path, fourcc, fps, (actual_width, actual_height))
            else:
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.frame_count = 0
        self.detection_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                self.detection_count += 1
                
                # Apply light smoothing
                smoothed_landmarks = self.smooth_landmarks(results.pose_landmarks)
                
                # Draw improved stick figure
                frame = self.draw_clean_stick_figure(frame, smoothed_landmarks)
                
                # Analyze form
                analysis = self.analyze_squat_form(smoothed_landmarks)

                # Add back posture analysis
                back_analysis = self.analyze_back_posture(smoothed_landmarks)

                # Draw spine visualization
                frame = self.draw_spine_visualization(frame, smoothed_landmarks, back_analysis)
                
                # Draw feedback (including back analysis)
                frame = self.draw_form_feedback(frame, analysis)
                frame, _ = self.draw_back_analysis_feedback(frame, back_analysis)

            else:
                # No detection
                cv2.putText(frame, 'No pose detected', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(frame, f'Frame: {self.frame_count} | Detections: {self.detection_count}', 
                           (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Improved Squat Analysis', frame)
            
            if output_path:
                out.write(frame)
            
            # Progress indicator
            if self.frame_count % 30 == 0:
                progress = (self.frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {progress:.1f}% - Detections: {self.detection_count}/{self.frame_count}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Total frames: {self.frame_count}")
        print(f"Successful detections: {self.detection_count}")
        print(f"Detection rate: {(self.detection_count/self.frame_count)*100:.1f}%")

