"""
phone_api.py - Flask API integration for the new phone detection functionality
This file provides Flask routes to integrate the new phone detection functionality from phone_detector_new.py
"""
from flask import Blueprint, request, jsonify, Response, current_app
from phone_detector_new import (
    add_camera, stop_camera, generate_frames, get_detections, 
    list_cameras, model, detect_phone_in_frame, save_fraud_detection_image
)
import base64
import cv2
import numpy as np
import os

# Create a Blueprint for phone detection API
phone_api = Blueprint('phone_api', __name__)

@phone_api.route('/add_camera', methods=['POST'])
def api_add_camera():
    """Add a new camera to the system using JSON body."""
    try:
        data = request.json
        camera_id = data.get('camera_id')
        source = data.get('source')
        confidence = data.get('confidence', 0.5)
        
        if not camera_id or not source:
            return jsonify({
                "success": False, 
                "message": "Missing required parameters: camera_id and source"
            })
        
        # Convert source to int if it's a digit (for webcam indexes)
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        success = add_camera(camera_id, source, confidence)
        
        if success:
            video_feed_url = f"/phone_api/video_feed/{camera_id}"
            return jsonify({
                "success": True, 
                "message": f"Camera {camera_id} added successfully",
                "video_feed_url": video_feed_url
            })
        else:
            return jsonify({
                "success": False, 
                "message": f"Failed to connect to camera source {source}"
            })
    except Exception as e:
        import traceback
        current_app.logger.error(f"Error adding camera: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "success": False, 
            "message": str(e)
        })

@phone_api.route('/stop_camera', methods=['POST'])
def api_stop_camera():
    """Stop and remove a camera from the system using JSON body."""
    try:
        data = request.json
        camera_id = data.get('camera_id')
        
        if not camera_id:
            return jsonify({
                "success": False, 
                "message": "Missing required parameter: camera_id"
            })
            
        success = stop_camera(camera_id)
        return jsonify({"success": success})
    except Exception as e:
        current_app.logger.error(f"Error stopping camera: {str(e)}")
        return jsonify({
            "success": False, 
            "message": str(e)
        })

@phone_api.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Stream processed video from a specific camera."""
    try:
        return Response(
            generate_frames(camera_id),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        current_app.logger.error(f"Error in video feed: {str(e)}")
        return jsonify({"error": str(e)})

@phone_api.route('/detections/<camera_id>')
def api_get_detections(camera_id):
    """Get phone detection results for a specific camera."""
    try:
        detections = get_detections(camera_id)
        return jsonify({"phoneDetections": detections})
    except Exception as e:
        current_app.logger.error(f"Error getting detections: {str(e)}")
        return jsonify({"error": str(e)})

@phone_api.route('/list_cameras')
def api_list_cameras():
    """List all active cameras."""
    try:
        cameras = list_cameras()
        return jsonify({
            "cameras": [
                {
                    "id": camera["id"],
                    "source": camera["source"],
                    "confidence": camera["confidence"],
                    "video_feed_url": f"/phone_api/video_feed/{camera['id']}"
                }
                for camera in cameras
            ]
        })
    except Exception as e:
        current_app.logger.error(f"Error listing cameras: {str(e)}")
        return jsonify({"error": str(e)})

@phone_api.route('/detect_in_image', methods=['POST'])
def detect_in_image():
    """
    Detect phone in an uploaded image
    This endpoint is compatible with the existing phone detection interface
    """
    try:
        if 'image' not in request.files:
            # Check if the image is in the form data as base64
            if 'image_base64' in request.form:
                image_b64 = request.form['image_base64']
                # Remove data URL prefix if present
                if ',' in image_b64:
                    image_b64 = image_b64.split(',')[1]
                # Decode base64 to image
                nparr = np.frombuffer(base64.b64decode(image_b64), np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                return jsonify({
                    "success": False,
                    "message": "No image provided"
                })
        else:
            # Read image from file upload
            file = request.files['image']
            nparr = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect phone in the image
        phone_detected, confidence = detect_phone_in_frame(model, image)
        
        # Save the image if a phone is detected
        saved_image_path = None
        if phone_detected:
            saved_image_path = save_fraud_detection_image(image, confidence)
        
        return jsonify({
            "success": True,
            "phone_detected": phone_detected,
            "confidence": float(confidence),
            "saved_image_path": saved_image_path
        })
    except Exception as e:
        current_app.logger.error(f"Error detecting in image: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        })

def init_app(app):
    """Initialize the phone API with the Flask app"""
    app.register_blueprint(phone_api, url_prefix='/phone_api')
