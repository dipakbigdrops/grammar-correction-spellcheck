"""
Fixed Grammar Correction Processor
All indentation and syntax errors resolved
"""
import os
import json
import re
import time
import base64
from io import BytesIO
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw
from typing import Tuple, List, Dict, Optional, Any
import logging
import torch

from app.config import settings
from app.robust_model_loader import load_robust_model, test_model_inference

logger = logging.getLogger(__name__)


class GrammarCorrectionProcessor:
    """Fixed grammar correction processor with singleton pattern"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once (singleton pattern)
        if GrammarCorrectionProcessor._initialized:
            return

        self.model = None
        self.tokenizer = None
        self.ocr_reader = None
        self.spell_checker = None
        self._load_model()
        self._initialize_ocr()
        self._initialize_spell_checker()

        GrammarCorrectionProcessor._initialized = True
        logger.info(" GrammarCorrectionProcessor initialized (singleton)")

    def _load_model(self):
        """Load model with ultimate robust error handling"""
        try:
            # Check if model path exists or if we should download from Hugging Face
            model_path = settings.MODEL_PATH
            model_id = getattr(settings, 'MODEL_ID', None)
            
            # If model path doesn't exist and MODEL_ID is set, download from Hugging Face
            if not os.path.exists(model_path) and model_id:
                logger.info(" Model path not found: %s, downloading from Hugging Face: %s", model_path, model_id)
                try:
                    from huggingface_hub import snapshot_download
                    hf_token = getattr(settings, 'HF_TOKEN', None)
                    os.makedirs(model_path, exist_ok=True)
                    snapshot_download(repo_id=model_id, local_dir=model_path, token=hf_token if hf_token else None)
                    logger.info(" Model downloaded successfully from Hugging Face")
                except Exception as download_error:
                    logger.error(" Failed to download model from Hugging Face: %s", download_error)
                    self.model = None
                    self.tokenizer = None
                    return
            
            if os.path.exists(model_path) or model_id:
                # Determine what to pass to load_robust_model
                # If model_path exists, use it; otherwise use MODEL_ID (HF repo ID)
                model_source = model_path if os.path.exists(model_path) else model_id
                logger.info(" Loading model from %s", model_source)

                # Get model info first (if model exists locally)
                if os.path.exists(model_path):
                    from app.robust_model_loader import get_model_info
                    model_info = get_model_info(model_path)
                    logger.info("Model info: %s", model_info)

                # Try to load with ultimate robust loader (can accept local path or HF repo ID)
                hf_token = getattr(settings, 'HF_TOKEN', None)
                self.model, self.tokenizer = load_robust_model(model_source, hf_token=hf_token)

                if self.model is not None and self.tokenizer is not None:
                    logger.info(" Model loaded successfully with ultimate robust loader")

                    # Test the model with a simple inference
                    try:
                        test_result = test_model_inference(self.model, self.tokenizer, "This is a test.")
                        logger.info(" Model test successful: '%s'", test_result)
                    except (RuntimeError, AttributeError) as test_e:
                        logger.warning("Model test failed but model loaded: %s", test_e)
                else:
                    logger.warning(" Model loading failed, using fallback")
                    self.model = None
                    self.tokenizer = None
            else:
                logger.warning(" Model path not found: %s and MODEL_ID not set", model_path)
                self.model = None
                self.tokenizer = None
        except (OSError, RuntimeError, ImportError) as e:
            logger.error(" Error loading model: %s", e)
            self.model = None
            self.tokenizer = None

    def _initialize_ocr(self):
        """Initialize OCR reader for text extraction from images"""
        try:
            import easyocr  # pylint: disable=import-outside-toplevel
            self.ocr_reader = easyocr.Reader(['en'])
            logger.info("OCR initialized")
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning("OCR not available: %s", e)
            self.ocr_reader = None

    def _initialize_spell_checker(self):
        """Initialize spell checker for catching spelling errors"""
        try:
            from spellchecker import SpellChecker  # pylint: disable=import-outside-toplevel
            self.spell_checker = SpellChecker(language='en')
            logger.info("Spell checker initialized")
        except (ImportError, Exception) as e:
            logger.warning("Spell checker not available: %s", e)
            self.spell_checker = None

    def is_ready(self) -> Dict[str, bool]:
        """Check readiness"""
        return {
            "model_loaded": self.model is not None,
            "ocr_available": self.ocr_reader is not None,
            "spell_checker_available": self.spell_checker is not None
        }

    def handle_input(self, input_source_path: str) -> Tuple[Optional[Any], str]:
        """
        Handle input file and determine its type.

        Args:
            input_source_path: Path to the input file

        Returns:
            Tuple of (content/path, input_type) or (None, error_type)
        """
        if not os.path.isfile(input_source_path):
            logger.error("File not found at %s", input_source_path)
            return None, 'file_not_found'

        file_extension = os.path.splitext(input_source_path)[1].lower()

        if file_extension in settings.ALLOWED_IMAGE_EXTENSIONS:
            return input_source_path, 'image'
        if file_extension in settings.ALLOWED_HTML_EXTENSIONS:
            # Try multiple encodings to handle different file formats
            encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1', 'cp1252']

            for encoding in encodings:
                try:
                    with open(input_source_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    logger.info("Successfully read HTML file with %s encoding", encoding)
                    return content, 'html'
                except UnicodeDecodeError:
                    continue
                except OSError as e:
                    logger.warning("Error reading HTML file with %s: %s", encoding, e)
                    continue

            # If all encodings fail, try reading as binary and decode with error handling
            try:
                with open(input_source_path, 'rb') as f:
                    raw_content = f.read()
                # Try to detect BOM and remove it
                if raw_content.startswith(b'\xff\xfe'):
                    content = raw_content[2:].decode('utf-16-le', errors='ignore')
                elif raw_content.startswith(b'\xfe\xff'):
                    content = raw_content[2:].decode('utf-16-be', errors='ignore')
                elif raw_content.startswith(b'\xef\xbb\xbf'):
                    content = raw_content[3:].decode('utf-8', errors='ignore')
                else:
                    content = raw_content.decode('utf-8', errors='ignore')
                logger.info("Successfully read HTML file with error handling")
                return content, 'html'
            except (OSError, IOError) as e:
                logger.error("Error reading HTML file with all methods: %s", e)
                return None, 'html_read_error'
        logger.error("Unsupported file type: %s", file_extension)
        return None, 'unknown_file_type'

    def extract_text(self, content: Any, input_type: str) -> Tuple[Any, Any]:
        """
        Extract text from image or HTML content.

        Args:
            content: Image path (str) or HTML content (str)
            input_type: Type of input ('image' or 'html')

        Returns:
            Tuple of (extracted_text, metadata)
        """
        if input_type == 'image':
            if not self.ocr_reader:
                logger.error("OCR reader not available")
                return [], []

            try:
                results = self.ocr_reader.readtext(content)
                extracted_texts = [item[1] for item in results]
                return extracted_texts, results
            except (OSError, ValueError, AttributeError) as e:
                logger.error("Error during OCR: %s", e)
                return [], []

        if input_type == 'html':
            # For HTML, we need to preserve the structure while extracting text for correction
            # Store the original HTML string to preserve formatting exactly
            original_html_string = content
            soup = BeautifulSoup(content, 'html.parser')

            # Extract ALL text content for grammar correction (not just specific elements)
            # This ensures we catch errors in any HTML element (td, th, etc.)
            # Use get_text() to extract all text, preserving structure for reconstruction
            extracted_text = soup.get_text(separator=' ', strip=False)
            
            # Clean up excessive whitespace while preserving structure
            # Replace multiple spaces/newlines with single space, but keep line breaks for readability
            import re
            extracted_text = re.sub(r'\s+', ' ', extracted_text)
            extracted_text = extracted_text.strip()

            # Return extracted text, soup object, and original HTML string for reconstruction
            return extracted_text, (soup, original_html_string)

        return None, None

    def correct_grammar(self, text: str) -> str:
        """Correct grammar with chunked processing and spell checking"""
        if not self.model or not self.tokenizer:
            logger.info("Model not available, using fallback correction")
            return self._fallback_correction(text)

        try:
            # Clean and prepare text for processing
            text = text.strip()
            if not text:
                return text

            # Step 1: Apply model correction
            # Check text length - if it's too long, process in chunks
            # Estimate tokens (rough approximation: 1 token ≈ 0.75 words)
            estimated_tokens = len(text.split()) * 1.33
            max_tokens_per_chunk = 100  # Leave some margin below 128 limit
            
            if estimated_tokens <= max_tokens_per_chunk:
                # Text is short enough, process normally
                model_corrected = self._correct_grammar_chunk(text)
            else:
                # Text is too long, process in chunks
                logger.info("Text is long (%d estimated tokens), processing in chunks", int(estimated_tokens))
                model_corrected = self._correct_grammar_chunked(text, max_tokens_per_chunk)
            
            # Step 2: Apply spell checking after model correction
            if self.spell_checker:
                spell_corrected = self._apply_spell_checking(model_corrected)
                return spell_corrected
            else:
                return model_corrected

        except Exception as e:
            logger.error("Error in correct_grammar: %s", e, exc_info=True)
            return self._fallback_correction(text)
    
    def _correct_grammar_chunk(self, text: str) -> str:
        """Correct grammar for a single chunk of text"""
        try:
            # Use the exact same logic as googlecolab.py
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)

            # Tokenize the input text (exactly like googlecolab.py)
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            # Generate the corrected text (exactly like googlecolab.py)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=128,
                    num_beams=5,  # Beam search for better quality
                    early_stopping=True
                )

            # Decode the generated IDs to text
            corrected_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # If result is empty, it's a model failure - use fallback
            if not corrected_text or corrected_text.strip() == "":
                logger.warning("Model returned empty result for chunk, using fallback")
                return self._fallback_correction(text)

            # Clean up the corrected text
            corrected_text = corrected_text.strip()
            return corrected_text

        except Exception as e:
            logger.error("Error correcting chunk: %s", e)
            return self._fallback_correction(text)
    
    def _correct_grammar_chunked(self, text: str, max_tokens_per_chunk: int = 100) -> str:
        """Process long text in chunks and combine results"""
        # Split text into sentences for better chunking
        # This preserves sentence boundaries which helps the model
        sentences = re.split(r'([.!?]\s+)', text)
        
        # Recombine sentences with their punctuation
        sentence_pairs = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence_pairs.append(sentences[i] + sentences[i + 1])
            else:
                sentence_pairs.append(sentences[i])
        if len(sentences) % 2 == 1:
            sentence_pairs.append(sentences[-1])
        
        # Group sentences into chunks that fit within token limit
        chunks = []
        current_chunk = ""
        
        for sentence in sentence_pairs:
            # Estimate tokens for current chunk + new sentence
            test_chunk = (current_chunk + " " + sentence).strip()
            estimated_tokens = len(test_chunk.split()) * 1.33
            
            if estimated_tokens <= max_tokens_per_chunk and current_chunk:
                # Add to current chunk
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.info("Split text into %d chunks for processing", len(chunks))
        
        # Process each chunk
        corrected_chunks = []
        for i, chunk in enumerate(chunks):
            logger.debug("Processing chunk %d/%d (length: %d chars)", i + 1, len(chunks), len(chunk))
            corrected_chunk = self._correct_grammar_chunk(chunk)
            corrected_chunks.append(corrected_chunk)
        
        # Combine corrected chunks
        corrected_text = " ".join(corrected_chunks)
        
        # Check if any corrections were made
        if corrected_text.strip() == text.strip():
            logger.info("No grammar errors found after chunked processing")
        else:
            logger.info("Grammar correction applied via chunked processing: '%s...' -> '%s...'", 
                       text[:50], corrected_text[:50])
        
        return corrected_text

    def _check_model_changes(self, original: str, corrected: str) -> bool:
        """Check if the model actually made changes"""
        return original.strip() != corrected.strip()
    
    def _apply_spell_checking(self, text: str) -> str:
        """Apply spell checking to catch errors the model missed"""
        if not self.spell_checker:
            return text
        
        try:
            # Split text into words while preserving punctuation and spacing
            # Use regex to find words (sequences of letters, including apostrophes)
            # Improved pattern to handle all word characters including uppercase
            words = re.findall(r"(\b[\w']+\b|\W+)", text)
            corrected_words = []
            corrections_made = 0
            
            for word in words:
                # Skip non-word tokens (punctuation, spaces, newlines)
                if not re.match(r"^[\w']+$", word):
                    corrected_words.append(word)
                    continue
                
                # Check spelling (case-insensitive)
                # Convert to lowercase for checking, but preserve original case
                word_lower = word.lower()
                # Remove apostrophes for checking (e.g., "don't" -> "dont")
                word_for_check = word_lower.replace("'", "")
                
                # Skip very short words (likely abbreviations or valid)
                if len(word_for_check) < 3:
                    corrected_words.append(word)
                    continue
                
                # Check if word is misspelled
                # Use unknown() method to check if word is misspelled
                # This is more reliable than checking dictionary membership
                unknown_words = self.spell_checker.unknown([word_for_check])
                is_misspelled = word_for_check in unknown_words
                
                if word_for_check and is_misspelled:
                    # Word is misspelled, get correction
                    correction = self.spell_checker.correction(word_for_check)
                    
                    if correction and correction != word_for_check and len(correction) > 0:
                        # Preserve original case
                        if word.isupper():
                            # All caps - keep all caps
                            correction = correction.upper()
                        elif word[0].isupper():
                            # Title case - capitalize first letter
                            correction = correction.capitalize()
                        # else: keep lowercase
                        
                        corrected_words.append(correction)
                        corrections_made += 1
                        logger.info("Spell checker correction: '%s' -> '%s'", word, correction)
                    else:
                        # No correction found, keep original
                        corrected_words.append(word)
                else:
                    # Word is correctly spelled
                    corrected_words.append(word)
            
            if corrections_made > 0:
                corrected_text = ''.join(corrected_words)
                logger.info("Spell checker applied %d corrections", corrections_made)
                return corrected_text
            else:
                logger.debug("Spell checker found no errors")
                return text
                
        except Exception as e:
            logger.error("Error in spell checking: %s", e, exc_info=True)
            return text

    def _fallback_correction(self, text: str) -> str:
        """Enhanced fallback corrections for common errors and OCR mistakes"""
        corrections = {
            # Common spelling mistakes
            r'\bgrammer\b': 'grammar',
            r'\bteh\b': 'the',
            r'\badn\b': 'and',
            r'\bthier\b': 'their',
            r'\brecieve\b': 'receive',
            r'\boccured\b': 'occurred',
            r'\bseperate\b': 'separate',
            r'\bdefinately\b': 'definitely',

            # Contractions
            r'\bdont\b': "don't",
            r'\bwont\b': "won't",
            r'\bcant\b': "can't",
            r'\bdoesnt\b': "doesn't",
            r'\bdidnt\b': "didn't",
            r'\bhavent\b': "haven't",
            r'\bhasnt\b': "hasn't",
            r'\bhadnt\b': "hadn't",
            r'\bisnt\b': "isn't",
            r'\bwasnt\b': "wasn't",
            r'\bwerent\b': "weren't",
            r'\bwouldnt\b': "wouldn't",
            r'\bcouldnt\b': "couldn't",
            r'\bshouldnt\b': "shouldn't",

            # OCR common mistakes (letter confusions)
            r'\b0\b': 'O',  # Zero confused with letter O
            r'\bl\b(?=[A-Z])': 'I',  # lowercase L confused with I
            r'\brn\b': 'm',  # rn confused with m
            r'\bvv\b': 'w',  # vv confused with w
        }

        corrected_text = text
        corrections_made = 0

        for pattern, replacement in corrections.items():
            new_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
            if new_text != corrected_text:
                corrections_made += 1
            corrected_text = new_text

        if corrections_made > 0:
            logger.info("Fallback correction applied %d fixes", corrections_made)

        return corrected_text

    def identify_corrections(self, original_text: str, corrected_text: str, context_words: int = 3) -> List[Dict[str, str]]:
        """
        Compares original and corrected text to identify changed words using sequence matching.
        Matches googlecolab.py exactly.
        """
        from difflib import Differ

        # Quick check: if texts are identical, no corrections needed
        if original_text.strip() == corrected_text.strip():
            logger.info("No corrections needed - texts are identical")
            return []

        # Tokenize including punctuation as separate tokens (exactly like googlecolab.py)
        original_tokens_with_sep = re.findall(r'(\b\w+\b|\W+)', original_text)
        corrected_tokens_with_sep = re.findall(r'(\b\w+\b|\W+)', corrected_text)

        # Create lists of only words for diffing
        original_words = [token.lower() for token in original_tokens_with_sep if re.fullmatch(r'\b\w+\b', token)]
        corrected_words = [token.lower() for token in corrected_tokens_with_sep if re.fullmatch(r'\b\w+\b', token)]

        differ = Differ()
        # Diff based on words only for identifying changes
        diff = list(differ.compare(original_words, corrected_words))

        corrections = []
        original_buffer = []
        corrected_buffer = []

        # Keep track of the index in the original_words and corrected_words lists
        original_word_index = 0
        corrected_word_index = 0

        for item in diff:
            code = item[0]
            token = item[2:]  # This is a word from original_words or corrected_words

            if code == '?':
                # Skip difference markers
                continue

            if code == '-':
                original_buffer.append(token)
                original_word_index += 1
            elif code == '+':
                corrected_buffer.append(token)
                corrected_word_index += 1
            elif code == ' ':
                # If tokens are the same, process any buffered changes before this
                while original_buffer or corrected_buffer:
                    orig = original_buffer.pop(0) if original_buffer else ''
                    corr = corrected_buffer.pop(0) if corrected_buffer else ''

                    # Only add to corrections if there's a change or non-empty insertion/deletion
                    if orig != corr or (orig == '' and corr != '') or (orig != '' and corr == ''):
                        # Find the index of the original word in the original_words list
                        try:
                            if orig:
                                # Find the index of the *last* occurrence of the original word in the original_words list before the current index
                                orig_index_in_words = original_word_index - len(original_buffer) - 1 if original_buffer else original_word_index - 1
                                orig_index_in_words = max(0, orig_index_in_words)

                                # Get original context words
                                original_context_start = max(0, orig_index_in_words - context_words)
                                original_context_end = min(len(original_words), orig_index_in_words + len([orig]) + context_words)
                                original_context = " ".join(original_words[original_context_start:original_context_end])
                            else:
                                # For insertions, context is based on the corrected text position
                                corr_index_in_words = corrected_word_index - len(corrected_buffer) - 1 if corrected_buffer else corrected_word_index - 1
                                corr_index_in_words = max(0, corr_index_in_words)

                                corrected_context_start = max(0, corr_index_in_words - context_words)
                                corrected_context_end = min(len(corrected_words), corr_index_in_words + len([corr]) + context_words)
                                original_context = " ".join(corrected_words[corrected_context_start:corrected_context_end])
                        except (IndexError, ValueError) as e:
                            logger.error("Error getting original context for %s: %s", orig, e)
                            original_context = ""

                        # Get corrected context words
                        try:
                            if corr:
                                corr_index_in_words = corrected_word_index - len(corrected_buffer) - 1 if corrected_buffer else corrected_word_index - 1
                                corr_index_in_words = max(0, corr_index_in_words)

                                corrected_context_start = max(0, corr_index_in_words - context_words)
                                corrected_context_end = min(len(corrected_words), corr_index_in_words + len([corr]) + context_words)
                                corrected_context = " ".join(corrected_words[corrected_context_start:corrected_context_end])
                            else:
                                # For deletions, context is based on the original text position
                                orig_index_in_words = original_word_index - len(original_buffer) - 1 if original_buffer else original_word_index - 1
                                orig_index_in_words = max(0, orig_index_in_words)

                                original_context_start = max(0, orig_index_in_words - context_words)
                                original_context_end = min(len(original_words), orig_index_in_words + len([orig]) + context_words)
                                corrected_context = " ".join(original_words[original_context_start:original_context_end])
                        except (IndexError, ValueError) as e:
                            logger.error("Error getting corrected context for %s: %s", corr, e)
                            corrected_context = ""

                        # Only add meaningful corrections (filter out empty or unchanged corrections)
                        if (orig.strip() != corr.strip() and
                            (orig.strip() != '' or corr.strip() != '') and
                            original_context.strip() != corrected_context.strip()):
                            corrections.append({
                                'original_word': orig.strip(),
                                'corrected_word': corr.strip(),
                                'original_context': original_context,
                                'corrected_context': corrected_context
                            })

                # Move indices forward for the matched token
                original_word_index += 1
                corrected_word_index += 1
                # Reset buffers
                original_buffer = []
                corrected_buffer = []

        # Process any remaining buffered changes at the end
        while original_buffer or corrected_buffer:
            orig = original_buffer.pop(0) if original_buffer else ''
            corr = corrected_buffer.pop(0) if corrected_buffer else ''

            if orig != corr or (orig == '' and corr != '') or (orig != '' and corr == ''):
                try:
                    if orig:
                        orig_index_in_words = original_word_index - len(original_buffer) - 1 if original_buffer else original_word_index - 1
                        orig_index_in_words = max(0, orig_index_in_words)

                        original_context_start = max(0, orig_index_in_words - context_words)
                        original_context_end = min(len(original_words), orig_index_in_words + len([orig]) + context_words)
                        original_context = " ".join(original_words[original_context_start:original_context_end])
                    else:
                        corr_index_in_words = corrected_word_index - len(corrected_buffer) - 1 if corrected_buffer else corrected_word_index - 1
                        corr_index_in_words = max(0, corr_index_in_words)

                        corrected_context_start = max(0, corr_index_in_words - context_words)
                        corrected_context_end = min(len(corrected_words), corr_index_in_words + len([corr]) + context_words)
                        original_context = " ".join(corrected_words[corrected_context_start:corrected_context_end])
                except (IndexError, ValueError) as e:
                    logger.error("Error getting original context for %s at end: %s", orig, e)
                    original_context = ""

                try:
                    if corr:
                        corr_index_in_words = corrected_word_index - len(corrected_buffer) - 1 if corrected_buffer else corrected_word_index - 1
                        corr_index_in_words = max(0, corr_index_in_words)

                        corrected_context_start = max(0, corr_index_in_words - context_words)
                        corrected_context_end = min(len(corrected_words), corr_index_in_words + len([corr]) + context_words)
                        corrected_context = " ".join(corrected_words[corrected_context_start:corrected_context_end])
                    else:
                        orig_index_in_words = original_word_index - len(original_buffer) - 1 if original_buffer else original_word_index - 1
                        orig_index_in_words = max(0, orig_index_in_words)

                        original_context_start = max(0, orig_index_in_words - context_words)
                        original_context_end = min(len(original_words), orig_index_in_words + len([orig]) + context_words)
                        corrected_context = " ".join(original_words[original_context_start:original_context_end])
                except (IndexError, ValueError) as e:
                    logger.error("Error getting corrected context for %s at end: %s", corr, e)
                    corrected_context = ""

                # Only add meaningful corrections (filter out empty or unchanged corrections)
                if (orig.strip() != corr.strip() and
                    (orig.strip() != '' or corr.strip() != '') and
                    original_context.strip() != corrected_context.strip()):
                    corrections.append({
                        'original_word': orig.strip(),
                        'corrected_word': corr.strip(),
                        'original_context': original_context,
                        'corrected_context': corrected_context
                    })

        # Final filter: remove meaningless corrections for website display
        cleaned_corrections = []
        for corr_dict in corrections:
            orig_word = corr_dict['original_word']
            corr_word = corr_dict['corrected_word']
            orig_context = corr_dict['original_context']
            corr_context = corr_dict['corrected_context']

            # Only keep corrections that are meaningful for website display:
            # 1. Words are actually different
            # 2. At least one word is not empty
            # 3. Contexts are different (indicating actual change)
            # 4. Both words are not empty (avoid empty corrections)
            if (orig_word != corr_word and
                (orig_word != '' or corr_word != '') and
                orig_context != corr_context and
                orig_word.strip() != '' and corr_word.strip() != ''):
                cleaned_corrections.append(corr_dict)

        logger.info("Filtered corrections: %d -> %d meaningful corrections", len(corrections), len(cleaned_corrections))

        return cleaned_corrections

    def reconstruct_with_highlighting(self, original_content: Any, input_type: str, corrected_text: str, corrections: List[Dict], original_ocr_results: Optional[List] = None) -> Optional[Any]:
        """
        Reconstructs the original content with highlighted corrections at the word level.
        Uses regex-based word matching for better accuracy (from googlecolab.py).
        """
        # Explicitly handle the case where no corrections are found
        if not corrections and input_type == 'image':
            logger.info("No corrections identified for image. Returning original image.")
            try:
                return Image.open(original_content).convert("RGB")
            except (OSError, IOError) as e:
                logger.error("Error loading original image for return: %s", e)
                return None
        if not corrections and input_type == 'html':
            logger.info("No corrections identified for %s. Returning original content.", input_type)
            return original_content

        # Proceed with highlighting only if corrections exist
        if input_type == 'image':
            if original_ocr_results is None:
                logger.error("Error: original_ocr_results is required for image input.")
                return None

            try:
                # Load the original image using the path
                img = Image.open(original_content).convert("RGB")
                draw = ImageDraw.Draw(img)

                # Create a set of original words that were corrected for quick lookup
                original_corrected_words_set = {
                    corr_dict['original_word'].lower()
                    for corr_dict in corrections
                    if corr_dict['original_word'] != corr_dict['corrected_word']
                }

                # Set a confidence threshold for highlighting
                confidence_threshold = 0.5

                # Iterate through the EasyOCR results (text blocks)
                for (bbox, text, confidence) in original_ocr_results:
                    # Only consider highlighting if confidence is above threshold
                    if confidence >= confidence_threshold:
                        # Get the bounding box coordinates as integers
                        x_coords = [int(p[0]) for p in bbox]
                        y_coords = [int(p[1]) for p in bbox]
                        x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

                        # Attempt word-level highlighting within the bounding box
                        block_text = text  # Use the original text from OCR
                        block_text_lower = block_text.lower()

                        # Iterate through the original words that were corrected
                        for original_word_lower in original_corrected_words_set:
                            # Find all occurrences of the original word within the block text
                            # Use regex to find whole words
                            for match in re.finditer(r'\b' + re.escape(original_word_lower) + r'\b', block_text_lower):
                                start_index = match.start()
                                word_length = len(match.group(0))

                                # Basic approximation for word position within the block
                                block_width = x2 - x1
                                char_width_approx = block_width / len(block_text) if len(block_text) > 0 else 0

                                word_x1 = x1 + (start_index * char_width_approx)
                                word_y1 = y1
                                word_x2 = word_x1 + (word_length * char_width_approx)
                                word_y2 = y2

                                # Draw a highlight (red rectangle border) around the approximate word bounding box
                                draw.rectangle([(word_x1, word_y1), (word_x2, word_y2)], outline='red', width=2)

                return img  # Return the PIL Image object

            except (OSError, IOError, ValueError) as e:
                logger.error("Error processing image for highlighting: %s", e)
                return None

        if input_type == 'html':
            try:
                # original_content is a tuple of (soup, original_html_string) from extract_text
                if isinstance(original_content, tuple) and len(original_content) == 2:
                    soup, html_string = original_content
                else:
                    # Fallback for old format
                    if hasattr(original_content, 'find_all'):
                        soup = original_content
                        html_string = str(soup)
                    else:
                        html_string = str(original_content)
                        soup = BeautifulSoup(html_string, 'html.parser')

                # Build a mapping of original words to their corrected versions
                word_corrections = {}
                for corr_dict in corrections:
                    orig_word = corr_dict.get('original_word', '')
                    corr_word = corr_dict.get('corrected_word', '')
                    if orig_word and orig_word != corr_word:
                        word_corrections[orig_word.lower()] = {
                            'original': orig_word,
                            'corrected': corr_word
                        }
                
                logger.info("HTML reconstruction: Processing %d corrections: %s", 
                           len(word_corrections), 
                           list(word_corrections.keys()))

                if not word_corrections:
                    # No corrections, return original HTML string
                    logger.warning("HTML reconstruction: No corrections to apply")
                    return html_string

                # DOM-based replacement: Traverse text nodes and insert real <u> Tag nodes
                # This approach is robust and handles all edge cases correctly
                from bs4 import NavigableString, Tag, Comment
                
                # Get all text nodes, excluding script, style, and comments
                text_nodes = []
                for element in soup.descendants:
                    if isinstance(element, NavigableString):
                        parent = element.parent
                        if parent:
                            parent_name = parent.name.lower() if parent.name else None
                            # Skip script, style, and comments
                            if parent_name not in ['script', 'style']:
                                # Check if parent is a comment or if element is inside a comment
                                is_comment = isinstance(parent, Comment)
                                # Also check if the string itself is a comment
                                if not is_comment and not isinstance(element, Comment):
                                    text_nodes.append(element)
                
                # Process each text node
                for text_node in text_nodes:
                    original_text = str(text_node)
                    if not original_text.strip():
                        continue
                    
                    # Find all words that need to be wrapped (with their positions)
                    # Use a set to track positions to avoid overlapping matches
                    word_matches = []
                    used_positions = set()
                    
                    for orig_lower, corr_data in word_corrections.items():
                        original_word = corr_data.get('original', '')
                        if not original_word:
                            continue
                        
                        # Escape special regex characters in the word
                        escaped_word = re.escape(original_word)
                        
                        # Use simple word boundary pattern - \b works well for most cases
                        # This is more reliable than the complex negative lookahead/lookbehind
                        word_pattern = r'\b' + escaped_word + r'\b'
                        
                        # Find all matches in this text node (case-insensitive)
                        for match in re.finditer(word_pattern, original_text, re.IGNORECASE | re.UNICODE):
                            start = match.start()
                            end = match.end()
                            
                            # Skip if this position range overlaps with a previous match
                            if any(start < prev_end and end > prev_start 
                                   for prev_start, prev_end, _ in word_matches):
                                continue
                            
                            # Get the actual word at this position (preserve original case)
                            actual_word = original_text[start:end]
                            
                            # Verify it's the same word (case-insensitive comparison)
                            if actual_word.lower() != orig_lower:
                                continue
                            
                            # Additional safety: verify it's not part of a larger word
                            # Check character before (if exists)
                            if start > 0:
                                char_before = original_text[start - 1]
                                # If it's a word character, skip (part of larger word)
                                if char_before.isalnum() or char_before == '_':
                                    continue
                            
                            # Check character after (if exists)
                            if end < len(original_text):
                                char_after = original_text[end]
                                # If it's a word character, skip (part of larger word)
                                if char_after.isalnum() or char_after == '_':
                                    continue
                            
                            word_matches.append((start, end, actual_word))
                            used_positions.add((start, end))
                    
                    if not word_matches:
                        continue
                    
                    # Sort matches by position (ascending) and remove any overlapping ones
                    word_matches.sort(key=lambda x: x[0])
                    
                    # Remove overlapping matches (keep first occurrence)
                    non_overlapping = []
                    for start, end, word in word_matches:
                        if not any(start < prev_end and end > prev_start 
                                  for prev_start, prev_end, _ in non_overlapping):
                            non_overlapping.append((start, end, word))
                    word_matches = non_overlapping
                    
                    # Build new content: split text and insert <u> tags
                    parent = text_node.parent
                    if not parent:
                        continue
                    
                    new_elements = []
                    last_pos = 0
                    
                    for start, end, word in word_matches:
                        # Add text before this match
                        if start > last_pos:
                            before_text = original_text[last_pos:start]
                            if before_text:
                                new_elements.append(NavigableString(before_text))
                        
                        # Create <u> tag with the word
                        u_tag = soup.new_tag('u')
                        u_tag.string = word
                        new_elements.append(u_tag)
                        
                        last_pos = end
                    
                    # Add remaining text after last match
                    if last_pos < len(original_text):
                        after_text = original_text[last_pos:]
                        if after_text:
                            new_elements.append(NavigableString(after_text))
                    
                    # Replace the original text node with new elements
                    if new_elements:
                        # Replace with first element
                        text_node.replace_with(new_elements[0])
                        # Insert remaining elements after the first
                        current = new_elements[0]
                        for elem in new_elements[1:]:
                            current.insert_after(elem)
                            current = elem
                
                # Convert soup back to string, preserving formatting
                # BeautifulSoup's str() preserves structure but may normalize some whitespace
                # For maximum preservation, we could use prettify with formatter=None,
                # but str() is sufficient and faster
                html_output = str(soup)
                
                # Ensure no escaped <u> tags (shouldn't happen with DOM-based approach, but verify)
                if '&lt;u&gt;' in html_output or '&lt;/u&gt;' in html_output:
                    logger.warning("Found escaped <u> tags in output, replacing")
                    html_output = html_output.replace('&lt;u&gt;', '<u>').replace('&lt;/u&gt;', '</u>')
                
                return html_output
                
            except (ValueError, AttributeError, re.error) as e:
                logger.error("HTML processing error: %s", e, exc_info=True)
                # Return original HTML string on error
                if isinstance(original_content, tuple) and len(original_content) == 2:
                    return original_content[1]
                return str(original_content) if original_content else None

        return None

    def generate_output(self, reconstructed_content: Any, input_type: str, corrections: List[Dict], output_dir: str = "/tmp") -> Tuple[Optional[str], str]:
        """
        Generate output - returns base64 for images, HTML string for HTML.

        Args:
            reconstructed_content: Processed content (Image or HTML)
            input_type: Type of input ('image' or 'html')
            corrections: List of correction dictionaries
            output_dir: Output directory (unused, kept for compatibility)

        Returns:
            Tuple of (content_output, json_output_string)
        """
        content_output = None

        if input_type == 'image':
            if isinstance(reconstructed_content, Image.Image):
                try:
                    # Convert image to base64 instead of saving to disk
                    buffered = BytesIO()
                    reconstructed_content.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    content_output = f"data:image/png;base64,{img_base64}"
                    logger.info("Image converted to base64 successfully")
                except (OSError, IOError) as e:
                    logger.error("Error converting image to base64: %s", e)
                    content_output = "Error converting image to base64"

        elif input_type == 'html':
            # reconstructed_content is already a properly formatted HTML string
            # with <u> tags and all original formatting preserved
            if isinstance(reconstructed_content, str):
                content_output = reconstructed_content
            elif hasattr(reconstructed_content, 'prettify'):
                # Fallback: convert soup to string (shouldn't happen with new logic)
                content_output = str(reconstructed_content)
            else:
                content_output = str(reconstructed_content) if reconstructed_content else None

        try:
            json_output_string = json.dumps(corrections, indent=4)
        except (TypeError, ValueError) as e:
            logger.error("Error generating JSON: %s", e)
            json_output_string = "[]"

        return content_output, json_output_string

    def process_input(self, input_source_path: str, output_dir: str = "/tmp") -> Dict[str, Any]:
        """
        Process input end-to-end.

        Args:
            input_source_path: Path to input file
            output_dir: Output directory (unused, kept for compatibility)

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()

        try:
            # 1. Handle input
            original_content, input_type = self.handle_input(input_source_path)

            if original_content is None:
                return {
                    "success": False,
                    "error": f"Failed to handle input: {input_type}",
                    "input_type": input_type
                }

            # 2. Extract text
            if input_type == 'image':
                extracted_texts, original_ocr_results = self.extract_text(original_content, input_type)
                text_to_correct = " ".join(extracted_texts) if extracted_texts else ""
                original_content_for_reconstruct = original_content
            elif input_type == 'html':
                extracted_text, soup_and_html = self.extract_text(original_content, input_type)
                text_to_correct = extracted_text if extracted_text else ""
                original_ocr_results = None
                # Pass both soup object and original HTML string for reconstruction
                original_content_for_reconstruct = soup_and_html
            else:
                return {
                    "success": False,
                    "error": "Unsupported input type",
                    "input_type": input_type
                }

            if not text_to_correct:
                        return {
                            "success": True,
                            "input_type": input_type,
                            "original_text": "",
                            "corrected_text": "",
                            "corrections": [],
                            "corrections_count": 0,
                            "output_file": None,
                    "processing_time_seconds": time.time() - start_time
                }

            # 3. Correct grammar
            corrected_text = self.correct_grammar(text_to_correct)

            # 4. Identify corrections
            if input_type == 'image':
                original_text_for_comparison = " ".join(extracted_texts) if extracted_texts else ""
            else:
                original_text_for_comparison = extracted_text

            corrections = self.identify_corrections(original_text_for_comparison, corrected_text)

            # 5. Reconstruct with highlighting
            reconstructed_content = self.reconstruct_with_highlighting(
                original_content_for_reconstruct,
                input_type,
                corrected_text,
                corrections,
                original_ocr_results=original_ocr_results if input_type == 'image' else None
            )

            # 6. Generate output
            content_output, json_output = self.generate_output(
                reconstructed_content,
                input_type,
                corrections,
                output_dir=output_dir
            )

            processing_time = time.time() - start_time

            return {
                "success": True,
                "input_type": input_type,
                "original_text": text_to_correct,
                "corrected_text": corrected_text,
                "corrections": corrections,
                "corrections_count": len(corrections),
                "output_content": content_output,  # Contains base64 image or HTML string
                "processing_time_seconds": round(processing_time, 2)
            }

        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Error in process_input: %s", e, exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "input_type": "unknown"
            }


# Global processor instance
_processor = None

def get_processor() -> GrammarCorrectionProcessor:
    """Get or create global processor instance"""
    global _processor
    if _processor is None:
        _processor = GrammarCorrectionProcessor()
    return _processor
