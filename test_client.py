"""
Example client script to test the Face Detection API
"""

import requests
import sys

def test_face_detection(image_path: str, api_url: str = "http://localhost:8000/detect-face"):
    """
    Test the face detection API with an image file.
    
    Args:
        image_path: Path to the image file
        api_url: URL of the API endpoint
    """
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path, f, 'image/jpeg')}
            response = requests.post(api_url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"   Has single face: {result['has_single_face']}")
            print(f"   Face count: {result['face_count']}")
            print(f"   Message: {result['message']}")
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
            return None
            
    except FileNotFoundError:
        print(f"❌ Error: Image file not found: {image_path}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Could not connect to API at {api_url}")
        print("   Make sure the API server is running!")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <path_to_image> [api_url]")
        print("Example: python test_client.py test_image.jpg")
        print("Example: python test_client.py test_image.jpg http://localhost:8000/detect-face")
        sys.exit(1)
    
    image_path = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000/detect-face"
    
    test_face_detection(image_path, api_url)

