# Face Detection API with LLM

A professional Python API that uses LLM vision models via OpenRouter to detect if an image contains exactly one human face. The API can detect faces even when they are covered by niqab, shemagh, masks, or other coverings.

## Features

- ✅ Detects exactly one human face in images
- ✅ Works with faces covered by niqab, shemagh, hijab, masks, etc.
- ✅ Concurrent request handling (multiple devices can use it simultaneously)
- ✅ Professional REST API with proper error handling
- ✅ Uses OpenRouter with GPT-4 Vision model (or other vision models) for accurate detection
- ✅ Returns `true` if exactly one face, `false` otherwise

## Installation

1. Clone or navigate to this directory:
```bash
cd api_facedetection_with_llm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenRouter API key:
   - Copy `.env.example` to `.env`
   - Get your API key from [OpenRouter](https://openrouter.ai/keys)
   - Add your OpenRouter API key to `.env`:
   ```
   OPENROUTER_API_KEY=your_actual_api_key_here
   VISION_MODEL=openai/gpt-4o
   ```
   
   **Note**: The `VISION_MODEL` must support image/vision input. Recommended models:
   - `openai/gpt-4o` (default, recommended)
   - `openai/gpt-4-vision-preview`
   - `anthropic/claude-3-opus`
   - `anthropic/claude-3.5-sonnet`
   - `google/gemini-pro-vision`

## Running the API

### Development Mode
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### 1. Root Endpoint
```
GET /
```
Returns API information and available endpoints.

### 2. Health Check
```
GET /health
```
Returns the health status of the API.

### 3. Face Detection (Main Endpoint)
```
POST /detect-face
```

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Image file (JPEG, PNG, etc.)

**Response:**
```json
{
  "has_single_face": true,
  "message": "Exactly one human face detected in the image",
  "face_count": 1
}
```

**Response when multiple faces:**
```json
{
  "has_single_face": false,
  "message": "Multiple faces detected (3 faces found). Expected exactly one face.",
  "face_count": 3
}
```

**Response when no faces:**
```json
{
  "has_single_face": false,
  "message": "No human faces detected in the image",
  "face_count": 0
}
```

## Usage Examples

### Using cURL
```bash
curl -X POST "http://localhost:8000/detect-face" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```

### Using Python
```python
import requests

url = "http://localhost:8000/detect-face"
files = {"file": open("path/to/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Using JavaScript/Fetch
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/detect-face', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## API Documentation

Once the server is running, you can access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## How It Works

1. The API receives an image file upload
2. The image is encoded to base64
3. The image is sent to OpenRouter (using GPT-4 Vision model) with specific instructions
4. The LLM analyzes the image and counts distinct human faces
5. The API returns `true` if exactly one face is detected, `false` otherwise

## Model Configuration

The API uses `openai/gpt-4o` by default through OpenRouter. You can change the model by setting the `VISION_MODEL` environment variable in your `.env` file. **Important**: The model must support vision/image input.

Available vision-capable models on OpenRouter:
- `openai/gpt-4o` (default, recommended)
- `openai/gpt-4-vision-preview`
- `anthropic/claude-3-opus`
- `anthropic/claude-3.5-sonnet`
- `google/gemini-pro-vision`

**Note**: Not all models on OpenRouter support image input. If you get an error "No endpoints found that support image input", the model you selected doesn't support vision. Use one of the models listed above.

## Important Notes

- The API can detect faces even when covered by:
  - Niqab (face veil)
  - Shemagh (head covering)
  - Masks
  - Hijab
  - Other coverings
- Maximum file size: 10MB
- Supported formats: JPEG, PNG, and other common image formats
- The API handles concurrent requests efficiently using FastAPI's async capabilities

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid file, empty file, file too large)
- `500`: Internal server error
- `503`: LLM service unavailable

## License

This project is provided as-is for use in your applications.

