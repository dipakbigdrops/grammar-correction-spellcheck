# API Usage Examples

## HTML Response Format

The `/process` endpoint supports returning HTML directly when processing HTML files.

### Using format=html Query Parameter

When you use `format=html` with an HTML input file, the API returns the corrected HTML directly with `Content-Type: text/html`, allowing it to be rendered in a browser.

#### Example: Process HTML and Get HTML Response

```bash
# Process HTML file and get HTML response
curl -X POST "http://localhost:8000/process?format=html" \
  -F "file=@example.html" \
  -H "Accept: text/html" \
  -o corrected.html

# Open in browser
# The response contains corrected HTML with <u> tags around corrected words
```

#### Example: Process HTML and Get JSON Response (default)

```bash
# Process HTML file and get JSON response
curl -X POST "http://localhost:8000/process?format=json" \
  -F "file=@example.html" \
  -H "Accept: application/json"

# Response:
# {
#   "task_id": "universal",
#   "status": "SUCCESS",
#   "result": {
#     "input_type": "html",
#     "output_content": "<html>...corrected HTML with <u> tags...</html>",
#     "corrections_count": 1
#   }
# }
```

### Using Preview Endpoint

When you process an HTML file with `format=html`, the response includes an `X-Preview-ID` header. You can use this ID to retrieve the HTML later using the preview endpoint.

#### Example: Get Preview ID and Retrieve HTML

```bash
# Step 1: Process HTML file and capture preview ID
RESPONSE=$(curl -i -X POST "http://localhost:8000/process?format=html" \
  -F "file=@example.html")

# Extract preview ID from X-Preview-ID header
PREVIEW_ID=$(echo "$RESPONSE" | grep -i "X-Preview-ID" | cut -d' ' -f2 | tr -d '\r')

# Step 2: Retrieve HTML using preview ID
curl "http://localhost:8000/process/preview/$PREVIEW_ID" \
  -o preview.html

# Open preview.html in browser
```

#### Example: Direct Browser Access

```bash
# Process file and get preview ID
PREVIEW_ID=$(curl -s -i -X POST "http://localhost:8000/process?format=html" \
  -F "file=@example.html" | grep -i "X-Preview-ID" | cut -d' ' -f2 | tr -d '\r')

# Open preview URL in browser
# http://localhost:8000/process/preview/$PREVIEW_ID
```

### Python Example

```python
import requests

# Process HTML file with format=html
with open('example.html', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/process?format=html',
        files={'file': f}
    )

# Get preview ID from headers
preview_id = response.headers.get('X-Preview-ID')

# Save HTML response
with open('corrected.html', 'w', encoding='utf-8') as f:
    f.write(response.text)

# Or retrieve later using preview endpoint
if preview_id:
    preview_response = requests.get(
        f'http://localhost:8000/process/preview/{preview_id}'
    )
    with open('preview.html', 'w', encoding='utf-8') as f:
        f.write(preview_response.text)
```

### JavaScript/Fetch Example

```javascript
// Process HTML file
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('/process?format=html', {
    method: 'POST',
    body: formData
});

// Get preview ID
const previewId = response.headers.get('X-Preview-ID');

// Get HTML content
const htmlContent = await response.text();

// Display in iframe or new window
const iframe = document.createElement('iframe');
iframe.srcdoc = htmlContent;
document.body.appendChild(iframe);

// Or open preview URL
if (previewId) {
    window.open(`/process/preview/${previewId}`, '_blank');
}
```

## Notes

- **Preview Expiration**: Previews are stored for 1 hour, after which they expire
- **HTML Format Only**: The `format=html` parameter only works with HTML input files
- **Content-Type**: When `format=html` is used, the response has `Content-Type: text/html`
- **Swagger UI**: The Swagger documentation shows both JSON and HTML response types for the `/process` endpoint

