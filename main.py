import os
import time
import base64
import json # Added for parsing JSON responses
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple # Added Tuple

from dotenv import load_dotenv
from openai import OpenAI
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from pydantic import BaseModel, Field, ValidationError
from deepdiff import DeepDiff
from flask import Flask, render_template, url_for, send_from_directory # Added Flask imports

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
# Global variable to store results for Flask app
comparison_results_for_flask = []

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_AI_VISION_ENDPOINT = os.getenv("AZURE_AI_VISION_ENDPOINT")
AZURE_AI_VISION_KEY = os.getenv("AZURE_AI_VISION_KEY")

# --- Pydantic Models for Recipe Structure ---

class Nutrition(BaseModel):
    servingSize: Optional[str] = Field(default=None, alias="servingSize")
    calories: Optional[str] = Field(default=None, alias="calories")
    fatContent: Optional[str] = Field(default=None, alias="fatContent")
    saturatedFatContent: Optional[str] = Field(default=None, alias="saturatedFatContent")
    transFatContent: Optional[str] = Field(default=None, alias="transFatContent")
    cholesterolContent: Optional[str] = Field(default=None, alias="cholesterolContent")
    sodiumContent: Optional[str] = Field(default=None, alias="sodiumContent")
    carbohydrateContent: Optional[str] = Field(default=None, alias="carbohydrateContent")
    fiberContent: Optional[str] = Field(default=None, alias="fiberContent")
    sugarContent: Optional[str] = Field(default=None, alias="sugarContent")
    proteinContent: Optional[str] = Field(default=None, alias="proteinContent")

    class Config:
        populate_by_name = True # Allows using alias for population
        # Ensure all fields from schema are present, even if optional in Pydantic for flexibility
        # The prompt to the LLM will request all fields as per the original schema's "required" list for nutrition

class RecipeInstructionItem(BaseModel):
    text: str

class Recipe(BaseModel):
    name: str
    cookTime: Optional[str] = Field(default=None, alias="cookTime")
    prepTime: Optional[str] = Field(default=None, alias="prepTime")
    totalTime: Optional[str] = Field(default=None, alias="totalTime")
    recipeYield: Optional[str] = Field(default=None, alias="recipeYield")
    recipeCategory: Optional[List[str]] = Field(default=None, alias="recipeCategory")
    recipeCuisine: Optional[List[str]] = Field(default=None, alias="recipeCuisine")
    recipeIngredient: List[str] = Field(alias="recipeIngredient")
    recipeInstructions: List[RecipeInstructionItem] = Field(alias="recipeInstructions")
    nutrition: Optional[Nutrition] = Field(default=None)

    class Config:
        populate_by_name = True

# --- File persistence functions ---

def save_results_to_file(results: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
    """
    Save OCR comparison results to a JSON file with metadata.
    Returns the filename of the saved file.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"ocr_results_{timestamp}.json"
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    filepath = results_dir / filename
    
    # Add metadata to the results
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(results),
            "openai_model_vision": "gpt-4.1-mini",
            "openai_model_structuring": "gpt-4o",
            "azure_vision_service": "Azure AI Vision",
            "schema_version": "1.0"
        },
        "results": results
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {filepath}")
    return str(filepath)

def load_results_from_file(filename: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load OCR comparison results from a JSON file.
    Returns the results list or None if file doesn't exist or is invalid.
    """
    filepath = Path(filename)
    if not filepath.exists():
        print(f"Results file not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if it's the new format with metadata
        if "results" in data and "metadata" in data:
            print(f"Loaded results from: {filepath}")
            print(f"Metadata: {data['metadata']}")
            return data["results"]
        else:
            # Assume it's the old format (just a list of results)
            print(f"Loaded results from: {filepath} (legacy format)")
            return data
            
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error loading results file {filepath}: {e}")
        return None

def find_latest_results_file() -> Optional[str]:
    """
    Find the most recent results file in the results directory.
    Returns the filename or None if no results files found.
    """
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    
    # Look for files matching the pattern ocr_results_*.json
    result_files = list(results_dir.glob("ocr_results_*.json"))
    if not result_files:
        return None
    
    # Sort by modification time and return the most recent
    latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)

def get_processed_images_from_results(results: List[Dict[str, Any]]) -> set:
    """
    Extract the set of image names that have been processed from results.
    """
    return {result["image"] for result in results if "image" in result}

def make_json_serializable(obj):
    """
    Convert DeepDiff and other non-JSON-serializable objects to JSON-serializable format.
    """
    # Handle None, basic types first
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Handle DeepDiff objects specifically
    if hasattr(obj, 'to_dict'):
        try:
            # DeepDiff objects have a to_dict method
            dict_obj = obj.to_dict()
            return make_json_serializable(dict_obj)
        except:
            # If to_dict fails, convert to string
            return str(obj)
    
    # Handle SetOrdered and other set-like objects
    if hasattr(obj, '__class__') and 'Set' in obj.__class__.__name__:
        return list(obj)
    
    # Handle regular sets
    if isinstance(obj, set):
        return list(obj)
    
    # Handle dictionaries
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Ensure keys are strings
            key = str(k) if not isinstance(k, str) else k
            result[key] = make_json_serializable(v)
        return result
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    
    # Handle objects with __dict__
    if hasattr(obj, '__dict__'):
        try:
            return make_json_serializable(obj.__dict__)
        except:
            return str(obj)
    
    # For anything else, test if it's already JSON serializable
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # As a last resort, convert to string
        return str(obj)

# --- Main functions and logic ---

def get_openai_structured_recipe_from_image(client: OpenAI, image_path: Path) -> Tuple[Optional[Recipe], float]:
    """
    Method 1: Sends an image to OpenAI (model with vision capabilities, e.g., gpt-4.1-mini)
    and asks for structured recipe data.
    Using "gpt-4.1-mini" as a placeholder for the user-specified gpt-4.1-mini, 
    as gpt-4.1-mini is a known model with strong vision capabilities.
    """
    print(f"Processing {image_path.name} with OpenAI Vision (gpt-4.1-mini)...")
    start_time = time.time()
    
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Constructing the prompt for JSON output based on Pydantic models
        # This is a simplified schema representation for the prompt.
        # For more complex schemas, providing a full JSON schema definition might be better.
        recipe_schema_prompt = Recipe.model_json_schema()

        response = client.chat.completions.create(
            model="gpt-4.1-mini", 
            messages=[
                {
                    "role": "system",
                    "content": "You analyze recipe images and extract structured recipe information. Carefully analyze this image for recipe information such as title, ingredients, instructions, cooking times, and servings. Extract all visible recipe information from the image. Format the information according to the required schema. Fix capitalization errors. Convert decimal values of ingredients to fractions when appropriate. Convert durations to hh:ss format if possible. Do the addition to provide an accurate total time for the recipe only if time estimates exist in the image."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Extract the recipe details from this image. Format the output as a JSON object that strictly adheres to the following JSON schema. Ensure all required fields are present:\n\n{json.dumps(recipe_schema_prompt, indent=2)}\n\nIf a value for an optional field is not found, omit the field or set it to null where appropriate based on the schema. For 'recipeIngredient' and 'recipeInstructions', ensure they are lists of strings and list of objects with a 'text' field respectively, even if empty."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}" # Assuming jpeg, adjust if other types are common
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000, # Adjust as needed
            response_format={"type": "json_object"}
        )
        
        raw_response_content = response.choices[0].message.content
        if not raw_response_content:
            print(f"Error (OpenAI Vision {image_path.name}): Empty response content.")
            return None, time.time() - start_time

        try:
            recipe_data = json.loads(raw_response_content)
            # Validate with Pydantic
            validated_recipe = Recipe(**recipe_data)
            return validated_recipe, time.time() - start_time
        except json.JSONDecodeError as e:
            print(f"Error (OpenAI Vision {image_path.name}): Failed to decode JSON: {e}")
            print(f"Raw response: {raw_response_content}")
            return None, time.time() - start_time
        except ValidationError as e:
            print(f"Error (OpenAI Vision {image_path.name}): Pydantic validation failed: {e}")
            print(f"Parsed data: {recipe_data}")
            return None, time.time() - start_time

    except Exception as e:
        print(f"Error (OpenAI Vision {image_path.name}): An unexpected error occurred: {e}")
        return None, time.time() - start_time

def get_azure_ocr_and_openai_structured_recipe(
    openai_client: OpenAI, azure_client: ImageAnalysisClient, image_path: Path
) -> Tuple[Optional[Recipe], float]:
    """
    Method 2: Uses Azure OCR to get text, then OpenAI (GPT-4o) to structure it.
    """
    print(f"Processing {image_path.name} with Azure OCR + OpenAI Structure (gpt-4o)...")
    total_start_time = time.time()
    azure_latency = 0.0
    openai_structuring_latency = 0.0
    
    # Step 1: Azure OCR
    try:
        print(f"  Azure OCR for {image_path.name}...")
        azure_ocr_start_time = time.time()
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        ocr_result = azure_client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.READ]
        )
        azure_latency = time.time() - azure_ocr_start_time

        if not (ocr_result.read and ocr_result.read.blocks):
            print(f"Error (Azure OCR {image_path.name}): No text found by Azure OCR.")
            return None, time.time() - total_start_time
        
        extracted_text = "\n".join([line.text for block in ocr_result.read.blocks for line in block.lines])
        print(f"  Azure OCR completed in {azure_latency:.2f}s. Extracted text length: {len(extracted_text)}")

    except Exception as e:
        print(f"Error (Azure OCR {image_path.name}): An unexpected error occurred: {e}")
        return None, time.time() - total_start_time

    # Step 2: OpenAI (GPT-4o) to structure the text
    try:
        print(f"  OpenAI GPT-4o structuring for {image_path.name}...")
        openai_structuring_start_time = time.time()
        
        recipe_schema_prompt = Recipe.model_json_schema()

        response = openai_client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {
                    "role": "system",
                    "content": "You are a cooking assistant. You receive recipes in an unstructured format. Follow these steps to produce a result: Step 1 - Parse all information relevant to the recipe out of the input and discard any noise. Step 2 - Don't change or add things to the recipe that weren't in the original text. Empty strings for properties are fine if they weren't in the original text. Step 3 - Fix capitalization errors. Step 4 - Convert decimal values of ingredients to fractions. Step 5 - Convert durations to hh:ss format if possible. Step 6 - Do the addition to provide an accurate total time for the recipe only if time estimates exist in the text. Step 7 - Verify you have followed the previous steps."
                },
                {
                    "role": "user",
                    "content": f"Convert the following raw text from OCR into a structured JSON recipe. Adhere strictly to this JSON schema. Ensure all required fields are present:\n\n{json.dumps(recipe_schema_prompt, indent=2)}\n\nIf a value for an optional field is not found, omit the field or set it to null where appropriate. For 'recipeIngredient' and 'recipeInstructions', ensure they are lists, even if empty.\n\nRaw text:\n{extracted_text}"
                }
            ],
            max_tokens=2000, # Adjust as needed
            response_format={"type": "json_object"}
        )
        openai_structuring_latency = time.time() - openai_structuring_start_time
        
        raw_response_content = response.choices[0].message.content
        if not raw_response_content:
            print(f"Error (OpenAI GPT-4o Structuring {image_path.name}): Empty response content.")
            return None, time.time() - total_start_time
            
        try:
            recipe_data = json.loads(raw_response_content)
            validated_recipe = Recipe(**recipe_data)
            total_latency = azure_latency + openai_structuring_latency
            print(f"  OpenAI GPT-4o structuring completed in {openai_structuring_latency:.2f}s.")
            return validated_recipe, total_latency
        except json.JSONDecodeError as e:
            print(f"Error (OpenAI GPT-4o Structuring {image_path.name}): Failed to decode JSON: {e}")
            print(f"Raw response: {raw_response_content}")
            return None, time.time() - total_start_time
        except ValidationError as e:
            print(f"Error (OpenAI GPT-4o Structuring {image_path.name}): Pydantic validation failed: {e}")
            print(f"Parsed data: {recipe_data}")
            return None, time.time() - total_start_time
            
    except Exception as e:
        print(f"Error (OpenAI GPT-4o Structuring {image_path.name}): An unexpected error occurred: {e}")
        return None, time.time() - total_start_time

def main():
    """
    Main function to orchestrate the comparison.
    """
    global comparison_results_for_flask
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="OCR Comparison Tool")
    parser.add_argument("--load-from", type=str, help="Load results from a specific file")
    parser.add_argument("--use-cache", action="store_true", help="Use the latest cached results if available")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocessing all images (ignore cache)")
    parser.add_argument("--save-to", type=str, help="Save results to a specific filename")
    parser.add_argument("--visualization-only", action="store_true", help="Skip processing and only run visualization")
    args = parser.parse_args()

    # Handle visualization-only mode
    if args.visualization_only:
        if args.load_from:
            results = load_results_from_file(args.load_from)
        elif args.use_cache:
            latest_file = find_latest_results_file()
            if latest_file:
                results = load_results_from_file(latest_file)
            else:
                print("No cached results found. Please run processing first.")
                return
        else:
            print("Visualization-only mode requires --load-from or --use-cache")
            return
        
        if results:
            comparison_results_for_flask = results
            print(f"Loaded {len(results)} results for visualization.")
            print("Starting Flask server for GUI...")
            print("Open http://127.0.0.1:5000/ in your browser to view results.")
            app.run(debug=True, use_reloader=False)
        return

    if not all([OPENAI_API_KEY, AZURE_AI_VISION_ENDPOINT, AZURE_AI_VISION_KEY]):
        print("Error: API keys or endpoint not configured. Please check your .env file.")
        return

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    azure_client = ImageAnalysisClient(
        endpoint=AZURE_AI_VISION_ENDPOINT,
        credential=AzureKeyCredential(AZURE_AI_VISION_KEY)
    )

    image_dir = Path("ocr-examples")
    if not image_dir.exists():
        print(f"Error: Image directory '{image_dir}' not found.")
        return

    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) # Add other extensions if needed

    if not image_files:
        print(f"No images found in '{image_dir}'.")
        return

    # Handle cache loading and determine which images to process
    existing_results = []
    images_to_process = image_files.copy()
    
    if not args.force_reprocess:
        # Try to load existing results
        if args.load_from:
            existing_results = load_results_from_file(args.load_from) or []
        elif args.use_cache:
            latest_file = find_latest_results_file()
            if latest_file:
                existing_results = load_results_from_file(latest_file) or []
        
        if existing_results:
            processed_images = get_processed_images_from_results(existing_results)
            images_to_process = [img for img in image_files if img.name not in processed_images]
            
            if images_to_process:
                print(f"Found {len(existing_results)} cached results. Processing {len(images_to_process)} new images.")
            else:
                print(f"All {len(image_files)} images already processed. Using cached results.")
                print("Use --force-reprocess to reprocess all images.")

    results = existing_results.copy()

    # Process only new images
    for image_path in images_to_process:
        print(f"\n--- Processing image: {image_path.name} ---")

        # Method 1: OpenAI Vision
        recipe_openai_vision, latency_openai_vision = get_openai_structured_recipe_from_image(openai_client, image_path)

        # Method 2: Azure OCR + OpenAI Structure
        recipe_azure_openai, latency_azure_openai = get_azure_ocr_and_openai_structured_recipe(
            openai_client, azure_client, image_path
        )

        diff = {}
        if recipe_openai_vision and recipe_azure_openai:
            # Convert Pydantic models to dicts for deepdiff
            # exclude_none=True helps in comparing only fields that have values
            dict1 = recipe_openai_vision.model_dump(exclude_none=True, by_alias=True)
            dict2 = recipe_azure_openai.model_dump(exclude_none=True, by_alias=True)
            diff = DeepDiff(dict1, dict2, ignore_order=True, verbose_level=0) # verbose_level=0 for concise output
        elif recipe_openai_vision:
            diff = {"method2_failed_to_produce_recipe": True}
        elif recipe_azure_openai:
            diff = {"method1_failed_to_produce_recipe": True}
        else:
            diff = {"both_methods_failed": True}

        results.append({
            "image": image_path.name,
            "method1_latency_seconds": latency_openai_vision,
            "method2_latency_seconds": latency_azure_openai,
            "method1_recipe": recipe_openai_vision.model_dump_json(indent=2, by_alias=True) if recipe_openai_vision else None,
            "method2_recipe": recipe_azure_openai.model_dump_json(indent=2, by_alias=True) if recipe_azure_openai else None,
            "differences": make_json_serializable(diff)
        })

        print(f"Latency (OpenAI Vision): {latency_openai_vision:.2f}s")
        print(f"Latency (Azure OCR + OpenAI Structure): {latency_azure_openai:.2f}s")
        if diff:
            print(f"Differences: {diff}")
        else:
            print("No differences found between the structured outputs.")

    # Save results to file
    if results:
        saved_file = save_results_to_file(results, args.save_to)
        print(f"\nResults saved to: {saved_file}")
    else:
        print("No results to save.")

    # Store results for Flask
    comparison_results_for_flask = results

    # Print summary to console as before
    print("\n\n--- Overall Results (Console Summary) ---")
    for res in results:
        print(f"\nImage: {res['image']}")
        print(f"  Method 1 Latency: {res['method1_latency_seconds']:.2f}s")
        print(f"  Method 2 Latency: {res['method2_latency_seconds']:.2f}s")
        # Limiting diff printing for console clarity
        diff_summary = str(res['differences'])
        if len(diff_summary) > 150:
            diff_summary = diff_summary[:150] + "..."
        print(f"  Differences: {diff_summary}")

    print("\nStarting Flask server for GUI...")
    print("Open http://127.0.0.1:5000/ in your browser to view results.")
    app.run(debug=True, use_reloader=False) # use_reloader=False to prevent running main() twice

@app.route('/')
def index():
    return render_template('overview.html', results=comparison_results_for_flask)

@app.route('/recipe/<image_name>')
def recipe_detail(image_name):
    result = next((r for r in comparison_results_for_flask if r['image'] == image_name), None)
    if result:
        # Parse JSON strings back to objects for easier handling in template if needed,
        # or pass as strings and handle in JS. For jsondiffpatch, strings are fine.
        method1_recipe_json = result['method1_recipe']
        method2_recipe_json = result['method2_recipe']
        
        # The DeepDiff object itself might not be directly JSON serializable for jsondiffpatch.
        # jsondiffpatch typically takes two JSON objects/strings.
        # We'll pass the original recipe JSONs and let jsondiffpatch compute the diff in the browser.
        return render_template('detail.html', 
                               result=result, 
                               image_name=image_name,
                               method1_json=method1_recipe_json,
                               method2_json=method2_recipe_json)
    return "Recipe not found", 404

@app.route('/ocr-examples/<filename>')
def serve_image(filename):
    return send_from_directory('ocr-examples', filename)

if __name__ == "__main__":
    # Ensure the ocr-examples directory exists for image serving
    if not Path("ocr-examples").exists():
        print("Error: 'ocr-examples' directory not found. Cannot serve images for GUI.")
    # Ensure templates directory exists
    if not Path("templates").exists():
        Path("templates").mkdir(parents=True, exist_ok=True)
        print("Created 'templates' directory.")
        # We'll create the actual template files in subsequent steps.
    main()
