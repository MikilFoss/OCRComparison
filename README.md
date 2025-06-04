# OCR Comparison Tool

This tool compares the performance of two OCR methods for structured recipe reading:

1.  **Method 1**: OpenAI Vision (e.g., gpt-4-vision-preview) - Direct image-to-structured-data
2.  **Method 2**: Azure OCR + OpenAI (e.g., gpt-4) - Two-step process: OCR extraction then structuring

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure API Keys**:
    Edit the `.env` file and add your credentials:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    AZURE_AI_VISION_ENDPOINT="your_azure_endpoint_here"
    AZURE_AI_VISION_KEY="your_azure_key_here"
    ```

3.  **Add Recipe Images**:
    Place your recipe images (JPG/PNG) in the `ocr-examples/` directory (you may need to create this directory if it doesn't exist).

## Usage

1.  **Run the Comparison**:
    To process all images in `ocr-examples/` and start the web server:
    ```bash
    python main.py
    ```

2.  **View Results**:
    - The script will process all images and display console output.
    - After processing, a web server will start.
    - Open `http://127.0.0.1:5000/` in your browser to view the GUI.

## GUI Features

### Overview Page
- Table showing all processed images
- Latency comparison for both methods
- Quick indication of whether differences were found
- Links to detailed view for each image

### Detail Page (per image)
- Original recipe image display
- Side-by-side JSON output comparison
- Visual diff highlighting
- Latency measurements for both methods

## Notes
- Press Ctrl+C in the terminal to stop the web server when done.
