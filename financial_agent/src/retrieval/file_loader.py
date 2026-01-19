import fitz  # PyMuPDF
from PIL import Image
import io
import base64
from typing import List, Dict, Any
from src.utils.llm import llm_client
from src.config import config
import os

class LocalFileLoader:
    def __init__(self):
        self.supported_exts = [".pdf"]

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Processes a local file, extracting text and analyzing images using GPT-4o.
        Returns a list of content blocks (text or image-description).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._process_pdf(file_path)
        else:
            print(f"Unsupported file extension: {ext}")
            return []

    def _process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        doc = fitz.open(file_path)
        content_blocks = []
        
        print(f"Processing PDF: {file_path} ({len(doc)} pages)")
        
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # We will collect image tasks
        image_tasks = []
        
        # 1. First Pass: Extract Text and Collect Images
        # Using tqdm for page iteration
        for page_num, page in enumerate(tqdm(doc, desc="Extracting Content")):
            # Text
            text = page.get_text()
            if text.strip():
                content_blocks.append({
                    "type": "text",
                    "content": text,
                    "page": page_num + 1
                })
            
            # Images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Filter small icons/logos (heuristic: < 15KB to skip more noise)
                if len(image_bytes) < 15360: 
                    continue
                    
                image_tasks.append({
                    "page_num": page_num + 1,
                    "image_bytes": image_bytes,
                    "xref": xref
                })

        print(f"Found {len(image_tasks)} potential financial images. Analyzing with Vision API...")
        
        # 2. Parallel Processing for GPT-4o Vision
        # Max workers limited to avoid rate limits
        if image_tasks:
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_img = {
                    executor.submit(self._analyze_image_with_gpt4o, task["image_bytes"]): task 
                    for task in image_tasks
                }
                
                for future in tqdm(as_completed(future_to_img), total=len(image_tasks), desc="Analyzing Images"):
                    task = future_to_img[future]
                    try:
                        description = future.result()
                        if description:
                            content_blocks.append({
                                "type": "image_description",
                                "content": f"[Image Description Page {task['page_num']}]: {description}",
                                "page": task['page_num'],
                                "original_image_xref": task['xref']
                            })
                    except Exception as e:
                        print(f"Error processing image on page {task['page_num']}: {e}")
                    
        return content_blocks

    def _analyze_image_with_gpt4o(self, image_bytes: bytes) -> str:
        """
        Sends image to GPT-4o for financial data extraction.
        """
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        prompt = """
        Analyze this image. If it contains a financial table, chart, or graph:
        1. Transcribe the key data points or table values into Markdown format.
        2. Summarize the trend or insight shown.
        
        If it is just a decorative image or logo, return "NO_FINANCIAL_DATA".
        """
        
        # We need to construct the payload for GPT-4o (Vision)
        # Using the OpenAI client directly from llm_client might need adjustment 
        # as analyze_text is text-only usually. 
        # accessing the client directly:
        
        try:
            response = llm_client.client.chat.completions.create(
                model=config.LLM_MODEL, # gpt-4o supports vision
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            result = response.choices[0].message.content
            if "NO_FINANCIAL_DATA" in result:
                return ""
            return result
        except Exception as e:
            print(f"GPT-4o Vision API Error: {e}")
            return ""

file_loader = LocalFileLoader()
