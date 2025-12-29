"""
Face Detection API using LLM Vision Model via OpenRouter
Detects if an image contains exactly one human face, including faces with coverings (niqab, shemagh, etc.)
"""

import os
import base64
import json
import re
import logging
from typing import Optional
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Detection API",
    description="API to detect if an image contains exactly one human face using LLM vision model via OpenRouter",
    version="1.0.0"
)

# Configure CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenRouter client (OpenRouter is compatible with OpenAI API format)
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
# Model to use - must support vision/image input
# Options: openai/gpt-4o, openai/gpt-4-vision-preview, anthropic/claude-3-opus, 
# anthropic/claude-3.5-sonnet, google/gemini-pro-vision
# Faster models: anthropic/claude-3-haiku, google/gemini-flash-1.5 (if available)
VISION_MODEL = os.getenv("VISION_MODEL", "openai/gpt-4o")
if not openrouter_api_key:
    logger.warning("OPENROUTER_API_KEY not found in environment variables")
    client = None
else:
    # OpenRouter uses OpenAI-compatible API, just change the base URL
    client = OpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1"
    )


class FaceDetectionResponse(BaseModel):
    """Response model for face detection"""
    has_single_face: bool
    message: str
    face_count: Optional[int] = None
    cropped_face: Optional[str] = None  # Base64 encoded cropped face image


def optimize_image(image_bytes: bytes, max_size: int = 512, quality: int = 75) -> bytes:
    """
    Optimize image by resizing if too large and compressing.
    This reduces payload size and speeds up API calls.
    
    Args:
        image_bytes: Original image bytes
        max_size: Maximum width/height in pixels (default 1024)
        quality: JPEG quality (1-100, default 85)
    
    Returns:
        Optimized image bytes
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        original_format = img.format
        
        # Convert RGBA to RGB if necessary (for JPEG)
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
            else:
                background.paste(img)
            img = background
        elif img.mode == 'P':
            # Palette mode - convert to RGB
            img = img.convert('RGB')
        
        # Resize if image is too large (use BILINEAR for good balance of speed/quality)
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
        
        # Save optimized image
        output = BytesIO()
        if original_format == 'PNG' and img.mode == 'RGB':
            # Save as JPEG for better compression
            img.save(output, format='JPEG', quality=quality, optimize=True)
        else:
            img.save(output, format=original_format or 'JPEG', quality=quality, optimize=True)
        
        optimized_bytes = output.getvalue()
        
        # Only use optimized if it's smaller (removed logging for speed)
        if len(optimized_bytes) < len(image_bytes):
            return optimized_bytes
        
        return image_bytes
    except Exception:
        # Silently fall back to original image on optimization failure
        return image_bytes


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode('utf-8')


def crop_face_from_image(image_bytes: bytes, crop_box: dict) -> Optional[bytes]:
    """
    Crop face from image using bounding box coordinates.
    
    Args:
        image_bytes: Original image bytes
        crop_box: Dictionary with normalized coordinates (0.0-1.0):
                  {"x_min": float, "y_min": float, "x_max": float, "y_max": float}
    
    Returns:
        Cropped image bytes, or None if cropping fails
    """
    try:
        # Open image
        img = Image.open(BytesIO(image_bytes))
        w, h = img.size
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(crop_box["x_min"] * w)
        y1 = int(crop_box["y_min"] * h)
        x2 = int(crop_box["x_max"] * w)
        y2 = int(crop_box["y_max"] * h)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Ensure valid box (x2 > x1, y2 > y1)
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid crop box: ({x1}, {y1}, {x2}, {y2})")
            return None
        
        # Crop the image
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_img.save("cropped_face.jpg")
        # Convert to bytes
        output = BytesIO()
        # Preserve original format if possible, otherwise use JPEG
        original_format = img.format or 'JPEG'
        if original_format in ['JPEG', 'JPG']:
            cropped_img.save(output, format='JPEG', quality=95)
        elif original_format == 'PNG':
            cropped_img.save(output, format='PNG')
        else:
            cropped_img.save(output, format='JPEG', quality=95)
        
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error cropping face: {str(e)}")
        return None


async def detect_faces_with_llm(image_bytes: bytes) -> dict:
    """
    Use LLM vision model to detect and count human faces in an image.
    Returns dict with 'face_count' and 'has_single_face' boolean.
    """
    try:
        # Optimize image for faster processing (resize large images, compress)
        optimized_image_bytes = optimize_image(image_bytes, max_size=512, quality=75)
        
        # Encode image to base64
        base64_image = encode_image_to_base64(optimized_image_bytes)
        
        # Determine image format (assuming JPEG or PNG)
        image_format = "jpeg"  # Default, can be enhanced to detect actual format
        
        # Prompt to detect faces and get bounding box coordinates
        user_prompt = """Count real human faces (including covered: niqab, shemagh, masks). For each face, provide the bounding box coordinates as normalized values (0.0 to 1.0) relative to image dimensions. Return JSON: {"face_count": N, "crop_box": {"x_min": 0.0-1.0, "y_min": 0.0-1.0, "x_max": 0.0-1.0, "y_max": 0.0-1.0}}. If multiple faces, provide the first/main face box. If no faces, omit crop_box."""

        # Check if client is initialized
        if client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenRouter API key not configured"
            )
        
        # Call OpenRouter Vision API with optimized settings
        # Remove system message and use only user message for faster processing
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=150,  # Increased to accommodate bounding box coordinates
            temperature=0.0,  # Fastest, most deterministic
            top_p=1.0  # Add top_p for faster token selection
        )
        
        # Parse the response (removed logging for speed)
        response_text = response.choices[0].message.content.strip()
        
        # Fast JSON parsing with fallback to number extraction
        face_count = 0
        crop_box = None
        try:
            # Try direct JSON parse first (fastest path)
            result = json.loads(response_text)
            face_count = int(result.get("face_count", 0))
            crop_box = result.get("crop_box", None)
        except (json.JSONDecodeError, ValueError, KeyError):
            # Fast fallback: extract first number found
            numbers = re.findall(r'\b(\d+)\b', response_text)
            face_count = int(numbers[0]) if numbers else 0
            # Try to extract crop_box from text if JSON parsing failed
            try:
                # Look for crop_box pattern in the text
                crop_box_match = re.search(r'"crop_box"\s*:\s*\{[^}]+\}', response_text)
                if crop_box_match:
                    crop_box_str = "{" + crop_box_match.group(0) + "}"
                    crop_box = json.loads(crop_box_str).get("crop_box")
            except:
                pass
        
        return {
            "face_count": face_count,
            "has_single_face": face_count == 1,
            "crop_box": crop_box,
            "raw_response": response_text
        }
        
    except Exception as e:
        # Check if it's an OpenRouter/API error
        error_str = str(e).lower()
        error_message = str(e)
        
        # Check for specific error about image input not supported
        if "no endpoints found that support image input" in error_message.lower() or "unsupported image input" in error_message.lower():
            logger.error(f"Model does not support image input: {error_message}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The selected model does not support image input. Please use a vision-capable model like 'openai/gpt-4o', 'openai/gpt-4-vision-preview', 'anthropic/claude-3-opus', or 'google/gemini-pro-vision'."
            )
        
        if "openrouter" in error_str or "openai" in error_str or "api" in error_str or "rate limit" in error_str or "404" in error_message:
            logger.error(f"OpenRouter API error: {error_message}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"LLM service error: {error_message}"
            )
        # Re-raise other exceptions to be handled by outer try-catch
        raise


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Face Detection API",
        "version": "1.0.0",
        "description": "Detect if an image contains exactly one human face using LLM vision model",
        "endpoints": {
            "/detect-face": "POST - Upload an image to detect faces",
            "/health": "GET - Health check endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "face-detection-api"
    }


@app.post("/detect-face", response_model=FaceDetectionResponse)
async def detect_face(file: UploadFile = File(...)):
    """
    Detect if the uploaded image contains exactly one human face.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON response with:
        - has_single_face: True if exactly one face detected, False otherwise
        - message: Descriptive message
        - face_count: Number of faces detected
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        # Check file size (limit to 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(image_bytes) > max_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds maximum allowed size of {max_size / (1024*1024)}MB"
            )
        
        # Removed logging for speed optimization
        
        # Detect faces using LLM
        detection_result = await detect_faces_with_llm(image_bytes)
        
        face_count = detection_result["face_count"]
        has_single_face = detection_result["has_single_face"]
        crop_box = detection_result.get("crop_box")
        
        # Crop face if bounding box is available
        cropped_face_base64 = None
        if crop_box and has_single_face:
            cropped_face_bytes = crop_face_from_image(image_bytes, crop_box)
            if cropped_face_bytes:
                cropped_face_base64 = encode_image_to_base64(cropped_face_bytes)
        
        # Create response message
        if face_count == 0:
            message = "No human faces detected in the image"
        elif face_count == 1:
            message = "Exactly one human face detected in the image"
        else:
            message = f"Multiple faces detected ({face_count} faces found). Expected exactly one face."
        
        return FaceDetectionResponse(
            has_single_face=has_single_face,
            message=message,
            face_count=face_count,
            cropped_face=cropped_face_base64
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        reload=True,
        workers=99  # For development; increase for production
    )

