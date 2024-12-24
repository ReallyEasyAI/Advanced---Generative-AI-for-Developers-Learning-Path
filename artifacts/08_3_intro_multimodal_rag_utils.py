"""
-----------------------------------------------------------------------------
This script demonstrates how to:
1. Load text and multimodal embedding models from Vertex AI.
2. Embed text and images (with optional contextual text) for similarity.
3. Process PDF documents:
    - Extract text by pages and chunk it with overlapping segments.
    - Extract images, save them as JPEG, and get their embeddings.
4. Generate text descriptions of images using a generative multimodal model.
5. Build Pandas DataFrames to store and manage text and image metadata.
6. Perform similarity searches for text-to-text or text-to-image queries.
7. Display results and citations in a user-friendly manner.
-----------------------------------------------------------------------------
"""

# ----------------------------- Imports --------------------------------------
# Below are all the necessary imports for this script.

import glob  # For finding all files matching certain patterns (like *.pdf)
import os    # For operating system dependent functionality (paths, dirs)
import time  # For adding delays or timestamps
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)  # Type hinting for clarity

# Third-party libraries
from IPython.display import display  # For displaying objects (like images) in notebooks
import PIL  # Image processing library (Pillow)
import fitz  # PyMuPDF for reading and handling PDF files
import numpy as np  # NumPy for numerical computations (vectors, arrays, etc.)
import pandas as pd  # Pandas for data manipulation and DataFrames
import requests  # For making HTTP requests (fetching images from URLs)

# Vertex AI imports
from vertexai.generative_models import (
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
    Image,  # This is a Vertex AI Image, not the same as PIL.Image
)
from vertexai.language_models import TextEmbeddingModel  # Vertex AI text embeddings
from vertexai.vision_models import (
    Image as VisionModelImage,  # Renamed to avoid confusion with PIL.Image
)
from vertexai.vision_models import MultiModalEmbeddingModel  # For image embeddings

# ----------------------------------------------------------------------------
#       LOADING MODELS FOR EMBEDDINGS
# ----------------------------------------------------------------------------

# Create an instance of a text embedding model from Vertex AI
text_embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

# Create an instance of a multimodal embedding model (for images) from Vertex AI
multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained(
    "multimodalembedding"
)

# ----------------------------------------------------------------------------
#       FUNCTIONS FOR GETTING TEXT AND IMAGE EMBEDDINGS
# ----------------------------------------------------------------------------

def get_text_embedding_from_text_embedding_model(
    text: str,
    return_array: Optional[bool] = False,
) -> Union[List[float], np.ndarray]:
    """
    Generates a numerical text embedding from a provided text input 
    using a Vertex AI text embedding model.

    Args:
        text (str): The input text string to be embedded.
        return_array (bool, optional): If True, returns the embedding as a 
            NumPy array. If False, returns it as a list of floats.
            Defaults to False.

    Returns:
        Union[List[float], np.ndarray]:
            A 768-dimensional vector representation of the input text.
            The format (list or NumPy array) depends on the 
            'return_array' parameter.

    Explanation for Beginners:
        - A "text embedding" is basically turning your text into a list (or array)
            of numbers that a computer can work with for tasks like searching 
            or comparing how similar two pieces of text are.
        - 'text_embedding_model.get_embeddings([text])' returns embeddings 
            for a list of text inputs. In our case, we just have one text input,
            so we wrap it in a list.
    """
    # Ask the Vertex AI text embedding model to generate embeddings 
    # for the given text. The model returns a list of embeddings (one per input).
    embeddings = text_embedding_model.get_embeddings([text])

    # We only have one embedding because we only passed one text, 
    # so we take the first one from the returned list.
    text_embedding = [embedding.values for embedding in embeddings][0]

    # If the user wants the embedding as a NumPy array, we convert 
    # the list of floats into a NumPy array.
    if return_array:
        text_embedding = np.fromiter(text_embedding, dtype=float)

    # Return the 768-dimensional embedding (list or array).
    return text_embedding


def get_image_embedding_from_multimodal_embedding_model(
    image_uri: str,
    embedding_size: int = 512,
    text: Optional[str] = None,
    return_array: Optional[bool] = False,
) -> Union[List[float], np.ndarray]:
    """
    Extracts an image embedding from a Vertex AI multimodal embedding model.
    Optionally uses some contextual text to refine the embedding.

    Args:
        image_uri (str): The local file path or a URL pointing to the image to process.
        embedding_size (int, optional): The desired dimensionality of the output 
            embedding. Defaults to 512.
        text (str, optional): Additional text context that can be used by 
            the model to adjust the embedding. Defaults to None.
        return_array (bool, optional): If True, returns the embedding as a NumPy array.
            Otherwise, returns it as a list of floats. Defaults to False.

    Returns:
        Union[List[float], np.ndarray]:
            A vector representing the image's features. Its size (dimension)
            depends on 'embedding_size'.

    Explanation for Beginners:
        - A "multimodal embedding" means the model can handle both images and text.
        - We can optionally supply some text to help the model focus on certain 
            features of the image.
    """
    # Load the image using VisionModelImage from the Vertex AI library
    # (not the same as PIL.Image).
    image = VisionModelImage.load_from_file(image_uri)

    # Get embeddings from the multimodal model; we pass in the image
    # and optionally some text context.
    embeddings = multimodal_embedding_model.get_embeddings(
        image=image,
        contextual_text=text,  # If you provided any text
        dimension=embedding_size,  # The dimension we want (128, 256, 512, etc.)
    )

    # The model returns an object containing separate embeddings for images and text.
    # Here, we're only interested in the image embedding.
    image_embedding = embeddings.image_embedding

    # Convert to NumPy array if that's what the user wants.
    if return_array:
        image_embedding = np.fromiter(image_embedding, dtype=float)

    return image_embedding


# ----------------------------------------------------------------------------
#       HELPER FUNCTION TO LOAD IMAGE BYTES FROM LOCAL OR WEB PATH
# ----------------------------------------------------------------------------

def load_image_bytes(image_path: str) -> bytes:
    """
    Loads an image from either a URL or local file path into bytes format.

    Args:
        image_path (str): URL or local file path to the image.

    Raises:
        ValueError: If 'image_path' is empty or None.

    Returns:
        bytes: The raw bytes of the image.

    Explanation for Beginners:
        - A URL starts with 'http://' or 'https://'. If the path doesn't start
            like that, we assume it's a local file path on your computer.
        - We read the image content in 'rb' mode, which means 'read bytes'.
    """
    # Check if the path is provided; if not, raise an error.
    if not image_path:
        raise ValueError("image_path must be provided.")

    # If the path looks like a link (begins with http or https), 
    # then fetch the image from the web.
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path, stream=True)
        if response.status_code == 200:
            return response.content

    # Otherwise, treat it as a local file path and read it in 'rb' mode.
    return open(image_path, "rb").read()


# ----------------------------------------------------------------------------
#       PDF HANDLING FUNCTION
# ----------------------------------------------------------------------------

def get_pdf_doc_object(pdf_path: str) -> Tuple[fitz.Document, int]:
    """
    Opens a PDF file using fitz (PyMuPDF) and returns the document 
    object along with the number of pages.

    Args:
        pdf_path (str): The path to the PDF file on your system.

    Returns:
        Tuple[fitz.Document, int]: A 2-tuple where:
            - The first element is the opened PDF Document object.
            - The second element is the total number of pages in the PDF.

    Raises:
        FileNotFoundError: If the PDF path is invalid or file not found.

    Explanation for Beginners:
        - fitz is the library behind PyMuPDF which lets us read PDF files.
        - 'fitz.open(pdf_path)' opens the PDF.
        - 'len(doc)' tells us how many pages the PDF has.
    """
    # Open the PDF file
    doc: fitz.Document = fitz.open(pdf_path)

    # Get the total number of pages in the PDF file
    num_pages: int = len(doc)

    return doc, num_pages


# ----------------------------------------------------------------------------
#       CLASS FOR COLOR PRINTING (USEFUL FOR TERMINAL OUTPUT)
# ----------------------------------------------------------------------------

class Color:
    """
    This class defines a set of color codes that can be used to print text 
    in different colors in a terminal. Great for highlighting or making 
    output more readable.

    Explanation for Beginners:
        - These are special character sequences that most terminals understand
            to render colored text.
    """
    PURPLE: str = "\033[95m"
    CYAN: str = "\033[96m"
    DARKCYAN: str = "\033[36m"
    BLUE: str = "\033[94m"
    GREEN: str = "\033[92m"
    YELLOW: str = "\033[93m"
    RED: str = "\033[91m"
    BOLD: str = "\033[1m"
    UNDERLINE: str = "\033[4m"
    END: str = "\033[0m"


# ----------------------------------------------------------------------------
#       FUNCTIONS FOR TEXT CHUNKING AND EMBEDDINGS
# ----------------------------------------------------------------------------

def get_text_overlapping_chunk(
    text: str, 
    character_limit: int = 1000, 
    overlap: int = 100
) -> Dict[int, str]:
    """
    Breaks a text document into chunks of a specified size, with a certain overlap 
    between chunks to preserve context between consecutive chunks.

    Args:
        text (str): The text document to chunk.
        character_limit (int, optional): Maximum characters per chunk. Defaults to 1000.
        overlap (int, optional): Number of overlapping characters between chunks. 
            Defaults to 100.

    Returns:
        Dict[int, str]:
            A dictionary where the keys are chunk numbers (1, 2, 3, ...) 
            and the values are the corresponding text chunks.

    Raises:
        ValueError: If 'overlap' is greater than 'character_limit'.

    Explanation for Beginners:
        - Sometimes you want to process a large piece of text in segments
            (chunks) but keep a bit of overlap so you don't lose context
            between those segments.
    """
    if overlap > character_limit:
        raise ValueError("Overlap cannot be larger than character limit.")

    # We will store chunk_number -> text_chunk in this dictionary.
    chunked_text_dict = {}

    # This variable helps us label chunks incrementally (1, 2, 3...).
    chunk_number = 1

    # We move through the text in steps of (character_limit - overlap),
    # ensuring the last chunk is cut properly if we go past the end of the text.
    for i in range(0, len(text), character_limit - overlap):
        end_index = min(i + character_limit, len(text))
        chunk = text[i:end_index]

        # For safety, we remove any non-ASCII characters that might cause issues.
        # Then we decode to "utf-8" ignoring problematic chars.
        chunked_text_dict[chunk_number] = chunk.encode(
            "ascii", "ignore"
        ).decode("utf-8", "ignore")

        # Move to the next chunk number.
        chunk_number += 1

    return chunked_text_dict


def get_page_text_embedding(text_data: Union[Dict[int, str], str]) -> Dict:
    """
    Generates embeddings for text data. If 'text_data' is a dictionary,
    we assume it is chunked text. Otherwise, we assume it's a single string.

    Args:
        text_data (Union[Dict[int, str], str]): Either:
            - A dictionary of pre-chunked text (chunk_number -> text).
            - A single text string.

    Returns:
        Dict:
            If 'text_data' is a dict, returns chunk_number -> embedding.
            If 'text_data' is a single string, returns "text_embedding" -> embedding.

    Explanation for Beginners:
        - This function figures out if it has a whole text or chunks.
            Then it calls our text embedding function for each chunk 
            or the entire text.
    """
    embeddings_dict = {}

    # If there's no text to embed, return an empty dictionary.
    if not text_data:
        return embeddings_dict

    # Check if text_data is a dictionary (implying multiple chunks).
    if isinstance(text_data, dict):
        # For each chunk in the dictionary, generate an embedding.
        for chunk_number, chunk_value in text_data.items():
            text_embed = get_text_embedding_from_text_embedding_model(
                text=chunk_value
            )
            embeddings_dict[chunk_number] = text_embed
    else:
        # If it's just a single string, embed the entire text.
        text_embed = get_text_embedding_from_text_embedding_model(text=text_data)
        embeddings_dict["text_embedding"] = text_embed

    return embeddings_dict


def get_chunk_text_metadata(
    page: fitz.Page,
    character_limit: int = 1000,
    overlap: int = 100,
    embedding_size: int = 128,
) -> Tuple[str, Dict, Dict, Dict]:
    """
    Extracts text from a given PDF page, chunks it, and generates embeddings 
    for both the entire page text and each chunk.

    Args:
        page (fitz.Page): A PyMuPDF page object to process.
        character_limit (int, optional): Max characters per chunk. Defaults to 1000.
        overlap (int, optional): Overlapping characters between chunks. Defaults to 100.
        embedding_size (int, optional): The size of the embedding vector. 
            Defaults to 128. (Note: Not used in this function directly.)

    Returns:
        Tuple[str, Dict, Dict, Dict]:
            1) The extracted page text as a string.
            2) A dict with a key "text_embedding" (the page-wide embedding).
            3) A dict of chunk_number -> chunk_text.
            4) A dict of chunk_number -> chunk_embedding.

    Raises:
        ValueError: If 'overlap' is larger than 'character_limit'.

    Explanation for Beginners:
        - We first convert all text to ASCII to avoid weird characters.
        - We generate one big embedding for the whole page.
        - Then we chunk the text and generate embeddings for each chunk too.
    """
    if overlap > character_limit:
        raise ValueError("Overlap cannot be larger than character limit.")

    # Extract text from the PDF page, removing non-ASCII characters.
    text: str = page.get_text().encode("ascii", "ignore").decode("utf-8", "ignore")

    # Get the embedding for the entire page text (un-chunked).
    page_text_embeddings_dict: Dict = get_page_text_embedding(text)

    # Chunk the text into smaller pieces (with overlap).
    chunked_text_dict: Dict = get_text_overlapping_chunk(text, character_limit, overlap)

    # Get embeddings for each chunk.
    chunk_embeddings_dict: Dict = get_page_text_embedding(chunked_text_dict)

    # Return a tuple containing all data (page text, page embedding, chunks, 
    # chunk embeddings).
    return text, page_text_embeddings_dict, chunked_text_dict, chunk_embeddings_dict


# ----------------------------------------------------------------------------
#       FUNCTION TO EXTRACT AND SAVE IMAGES FROM A PDF, THEN LOAD FOR GEMINI
# ----------------------------------------------------------------------------

def get_image_for_gemini(
    doc: fitz.Document,
    image: Tuple[Any, ...],
    image_no: int,
    image_save_dir: str,
    file_name: str,
    page_num: int,
) -> Tuple[Optional[Image], Optional[str]]:
    """
    Extracts an image from a PDF and saves it to a JPEG file, 
    taking care of color space conversions. Finally, it loads 
    it as a Vertex AI Image object for further processing.

    Args:
        doc (fitz.Document): The PDF document object.
        image (Tuple[Any, ...]): A tuple containing info about the image in the PDF.
        image_no (int): An index for enumerating images found on the page.
        image_save_dir (str): Directory where we should save the extracted images.
        file_name (str): The base name of the PDF file (used in naming the image).
        page_num (int): The page number the image came from.

    Returns:
        Tuple[Optional[Image], Optional[str]]:
            1) A Vertex AI Image object of the saved image.
            2) The file path to the saved image.

        If an error occurs, returns (None, None).

    Explanation for Beginners:
        - PDF images might use different color spaces (Gray, RGB, CMYK, etc.).
            We make sure to convert them to RGB if needed.
        - We then create a unique JPEG filename and save the image to disk.
        - Finally, we load that JPEG back into a Vertex AI Image for embeddings.
    """
    try:
        # xref is a reference ID for the image within the PDF.
        xref = image[0]

        # Create a Pixmap, which is essentially the image object inside the PDF.
        pix = fitz.Pixmap(doc, xref)

        # Convert to RGB color space if not already in an acceptable format.
        if pix.colorspace not in (fitz.csGRAY, fitz.csRGB, fitz.csCMYK):
            pix = fitz.Pixmap(fitz.csRGB, pix)

        # Build a name for saving this image as JPEG.
        image_name = (
            f"{image_save_dir}/{file_name}_image_{page_num}_{image_no}_{xref}.jpeg"
        )

        # Make sure the save directory exists.
        os.makedirs(image_save_dir, exist_ok=True)

        # Save the extracted image as a JPEG.
        pix.save(image_name)

        # Load it back as a Vertex AI Image object.
        image_for_gemini = Image.load_from_file(image_name)

        return image_for_gemini, image_name

    except Exception as e:
        print(f"Unexpected error processing image: {e}")
        return None, None


# ----------------------------------------------------------------------------
#       FUNCTION TO GENERATE CONTENT (TEXT) WITH A GENERATIVE MULTIMODAL MODEL
# ----------------------------------------------------------------------------

def get_gemini_response(
    generative_multimodal_model: Any,
    model_input: List[Union[str, Image]],
    stream: bool = True,
    generation_config: Optional[GenerationConfig] = GenerationConfig(
        temperature=0.2, max_output_tokens=2048
    ),
    safety_settings: Optional[dict] = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
) -> str:
    """
    Uses a generative multimodal model (like Vertex AI's Gemini) to generate text 
    based on an input list (which can include text strings and images).

    Args:
        generative_multimodal_model (Any): The model object capable of generating text.
        model_input (List[Union[str, Image]]): A list containing text and/or images.
        stream (bool, optional): Whether to generate the response in a streaming fashion
            or all at once. Defaults to True.
        generation_config (GenerationConfig, optional): Model parameters 
            (like temperature, token limits, etc.).
        safety_settings (dict, optional): Settings that determine how the model 
            handles sensitive or harmful content.

    Returns:
        str: The combined text generated by the model.

    Explanation for Beginners:
        - 'model_input' can be multiple items. For example:
            [ "Describe this image:", <Image Object> ]
        - The model will read these inputs, possibly interpret the image, 
            and generate text as output.
        - We collect all chunks of text if the model streams them and then combine 
            them into a single string.
    """
    # Generate the response in streaming or non-streaming mode.
    response = generative_multimodal_model.generate_content(
        model_input,
        generation_config=generation_config,
        stream=stream,
        safety_settings=safety_settings,
    )

    # We'll store the chunks of text here.
    response_list = []

    # For each chunk in the model's streaming response, append it to our list.
    for chunk in response:
        try:
            response_list.append(chunk.text)
        except Exception as e:
            print(
                "Exception occurred while calling Gemini. "
                "Something might be wrong with the content or safety settings. "
                "Error:",
                e,
            )
            response_list.append("Exception occurred")
            continue

    # Join all the text chunks into one final string.
    response = "".join(response_list)

    return response


# ----------------------------------------------------------------------------
#       FUNCTIONS TO BUILD DATAFRAMES FOR TEXT AND IMAGE METADATA
# ----------------------------------------------------------------------------

def get_text_metadata_df(
    filename: str, 
    text_metadata: Dict[Union[int, str], Dict]
) -> pd.DataFrame:
    """
    Builds a Pandas DataFrame from text metadata extracted from each page 
    of a PDF.

    Args:
        filename (str): The name of the document (PDF file).
        text_metadata (Dict[Union[int, str], Dict]): A dictionary where:
            - Keys are page numbers.
            - Values contain the text, chunked text, and embeddings for that page.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - file_name
            - page_num
            - text (the entire page text)
            - text_embedding_page (embedding for the entire page)
            - chunk_number (which chunk of the page)
            - chunk_text (the actual text chunk)
            - text_embedding_chunk (embedding for that chunk)

    Explanation for Beginners:
        - We iterate over each page in 'text_metadata'.
        - For each chunk in that page, we store all relevant info (page text, chunk text,
            embeddings) into a list of dictionaries. Then we turn that list into a DataFrame.
    """
    final_data_text: List[Dict] = []

    # Go through each page in the metadata
    for key, values in text_metadata.items():
        # Each page can have multiple text chunks
        for chunk_number, chunk_text in values["chunked_text_dict"].items():
            data: Dict = {}
            data["file_name"] = filename
            # Page numbers are zero-based in code, but humans read them as 1-based
            data["page_num"] = int(key) + 1
            data["text"] = values["text"]
            data["text_embedding_page"] = values["page_text_embeddings"]["text_embedding"]
            data["chunk_number"] = chunk_number
            data["chunk_text"] = chunk_text
            data["text_embedding_chunk"] = values["chunk_embeddings_dict"][chunk_number]
            final_data_text.append(data)

    # Convert the list of dicts to a Pandas DataFrame.
    return_df = pd.DataFrame(final_data_text)

    # Reset the DataFrame index so it goes 0,1,2,... 
    # instead of possibly skipping or repeating numbers.
    return_df = return_df.reset_index(drop=True)
    return return_df


def get_image_metadata_df(
    filename: str, 
    image_metadata: Dict[Union[int, str], Dict]
) -> pd.DataFrame:
    """
    Builds a Pandas DataFrame from image metadata extracted from each page 
    of a PDF.

    Args:
        filename (str): The name of the document (PDF file).
        image_metadata (Dict[Union[int, str], Dict]): A dictionary where:
            - Keys are page numbers.
            - Values contain data for each extracted image on that page.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - file_name
            - page_num
            - img_num
            - img_path
            - img_desc
            - mm_embedding_from_img_only
            - text_embedding_from_image_description

    Explanation for Beginners:
        - For each page, we might have multiple images. 
        - We collect each image's details (where it's saved, 
            what the model described, embeddings, etc.) into a list of dicts 
            and then build a DataFrame out of them.
    """
    final_data_image: List[Dict] = []

    # Loop through each page
    for key, values in image_metadata.items():
        # For each image on that page
        for _, image_values in values.items():
            data: Dict = {}
            data["file_name"] = filename
            data["page_num"] = int(key) + 1
            data["img_num"] = int(image_values["img_num"])
            data["img_path"] = image_values["img_path"]
            data["img_desc"] = image_values["img_desc"]
            data["mm_embedding_from_img_only"] = image_values["mm_embedding_from_img_only"]
            data["text_embedding_from_image_description"] = image_values[
                "text_embedding_from_image_description"
            ]
            final_data_image.append(data)

    return_df = pd.DataFrame(final_data_image).dropna()
    return_df = return_df.reset_index(drop=True)
    return return_df


# ----------------------------------------------------------------------------
#       MAIN FUNCTION TO EXTRACT METADATA FROM ALL PDFS IN A FOLDER
# ----------------------------------------------------------------------------

def get_document_metadata(
    generative_multimodal_model: Any,
    pdf_folder_path: str,
    image_save_dir: str,
    image_description_prompt: str,
    embedding_size: int = 128,
    generation_config: Optional[GenerationConfig] = GenerationConfig(
        temperature=0.2, max_output_tokens=2048
    ),
    safety_settings: Optional[dict] = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
    add_sleep_after_page: bool = False,
    sleep_time_after_page: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes all PDF files in a given folder to extract text and images, 
    build their embeddings, generate image descriptions, and store everything 
    in two Pandas DataFrames.

    Args:
        generative_multimodal_model (Any): A generative model (like Gemini) 
            that can process images and text prompts.
        pdf_folder_path (str): The folder containing PDF files to be processed.
        image_save_dir (str): Where to save extracted images.
        image_description_prompt (str): A text prompt to guide the model 
            in describing each image.
        embedding_size (int, optional): Dimensionality of the image embeddings. 
            Defaults to 128.
        generation_config (GenerationConfig, optional): Model generation settings.
        safety_settings (dict, optional): Safety thresholds for the model.
        add_sleep_after_page (bool, optional): Whether to add a delay after each page. 
            Defaults to False.
        sleep_time_after_page (int, optional): How many seconds to sleep if 
            'add_sleep_after_page' is True. Defaults to 2.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            1) text_metadata_df_final: A DataFrame containing text data, 
                chunked text, and their embeddings.
            2) image_metadata_df_final: A DataFrame containing image data, 
                descriptions, and their embeddings.

    Explanation for Beginners:
        - We'll look for every PDF file in 'pdf_folder_path'.
        - For each PDF, we open it and go page by page.
        - For each page, we extract text and chunk it, then get embeddings.
        - We also extract any images on that page, saving them as JPEGs, 
            generating a description, and also embeddings.
        - The final results are stored in two DataFrames: one for text, 
            one for images.
    """
    # We'll keep appending data from each PDF into these DataFrames.
    text_metadata_df_final = pd.DataFrame()
    image_metadata_df_final = pd.DataFrame()

    # Go through all PDF files in the specified folder.
    for pdf_path in glob.glob(pdf_folder_path + "/*.pdf"):
        print("\n\nProcessing the file: ---------------------------------", pdf_path, "\n")

        # Open the PDF and figure out how many pages it has.
        doc, num_pages = get_pdf_doc_object(pdf_path)

        # Just the PDF filename (without the folder path).
        file_name = pdf_path.split("/")[-1]

        # We'll store text and image data for each page in these dicts.
        text_metadata: Dict[Union[int, str], Dict] = {}
        image_metadata: Dict[Union[int, str], Dict] = {}

        # Loop over each page by its index (0 to num_pages-1).
        for page_num in range(num_pages):
            print(f"Processing page: {page_num + 1}")

            # Grab the PyMuPDF page object
            page = doc[page_num]

            # Extract text and chunk it, also get embeddings
            (
                text,
                page_text_embeddings_dict,
                chunked_text_dict,
                chunk_embeddings_dict,
            ) = get_chunk_text_metadata(page, embedding_size=embedding_size)

            # Store this page's text info in our dictionary.
            text_metadata[page_num] = {
                "text": text,
                "page_text_embeddings": page_text_embeddings_dict,
                "chunked_text_dict": chunked_text_dict,
                "chunk_embeddings_dict": chunk_embeddings_dict,
            }

            # Extract all images from this page
            images = page.get_images()
            image_metadata[page_num] = {}

            for image_no, image_info in enumerate(images):
                # image_no is index-based (0, 1, 2, ...)
                image_number = int(image_no + 1)  # Make it 1-based

                image_metadata[page_num][image_number] = {}

                # Extract the image and convert it into a Vertex AI Image
                # object we can feed to the generative model.
                image_for_gemini, image_name = get_image_for_gemini(
                    doc, image_info, image_no, image_save_dir, file_name, page_num
                )

                print(
                    f"Extracting image from page: {page_num + 1}, "
                    f"saved as: {image_name}"
                )

                # If something went wrong and there's no image, skip.
                if image_for_gemini is None:
                    continue

                # Generate an image description by feeding the image
                # and a prompt into the generative model.
                response = get_gemini_response(
                    generative_multimodal_model,
                    model_input=[image_description_prompt, image_for_gemini],
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=True,
                )

                # Get an embedding for the image (without context).
                image_embedding = get_image_embedding_from_multimodal_embedding_model(
                    image_uri=image_name,
                    embedding_size=embedding_size,
                )

                # Embed the text description of the image.
                image_description_text_embedding = get_text_embedding_from_text_embedding_model(
                    text=response
                )

                # Store all image metadata in our dictionary.
                image_metadata[page_num][image_number] = {
                    "img_num": image_number,
                    "img_path": image_name,
                    "img_desc": response,
                    "mm_embedding_from_img_only": image_embedding,
                    "text_embedding_from_image_description": image_description_text_embedding,
                }

            # If user requested to sleep after each page, do so here.
            if add_sleep_after_page:
                time.sleep(sleep_time_after_page)
                print(
                    "Sleeping for ",
                    sleep_time_after_page,
                    " sec before processing the next page "
                    "to avoid quota issues. You can disable it "
                    'by setting "add_sleep_after_page=False".'
                )

        # After processing all pages of this PDF, build the data frames
        text_metadata_df = get_text_metadata_df(file_name, text_metadata)
        image_metadata_df = get_image_metadata_df(file_name, image_metadata)

        # Append these frames to the final ones
        text_metadata_df_final = pd.concat(
            [text_metadata_df_final, text_metadata_df],
            axis=0
        )
        image_metadata_df_final = pd.concat(
            [
                image_metadata_df_final,
                image_metadata_df.drop_duplicates(subset=["img_desc"]),
            ],
            axis=0
        )

        # Reset the indices so they're sequential
        text_metadata_df_final = text_metadata_df_final.reset_index(drop=True)
        image_metadata_df_final = image_metadata_df_final.reset_index(drop=True)

    return text_metadata_df_final, image_metadata_df_final


# ----------------------------------------------------------------------------
#       HELPER FUNCTIONS FOR USER QUERIES AND SIMILARITY
# ----------------------------------------------------------------------------

def get_user_query_text_embeddings(user_query: str) -> np.ndarray:
    """
    Extracts text embeddings for the user's query text.

    Args:
        user_query (str): The user's question or search text.

    Returns:
        np.ndarray: The numerical embedding of that query text.

    Explanation for Beginners:
        - This is just a convenience wrapper around 'get_text_embedding_from_text_embedding_model'
            that always returns a NumPy array.
    """
    return get_text_embedding_from_text_embedding_model(user_query)


def get_user_query_image_embeddings(
    image_query_path: str, 
    embedding_size: int
) -> np.ndarray:
    """
    Extracts image embeddings for the user's query image.

    Args:
        image_query_path (str): The path or URL to the image query.
        embedding_size (int): The desired embedding size for the image.

    Returns:
        np.ndarray: The numerical embedding of that image.

    Explanation for Beginners:
        - If you're searching for images similar to this query image,
            you first convert your query image into an embedding as well.
    """
    return get_image_embedding_from_multimodal_embedding_model(
        image_uri=image_query_path, 
        embedding_size=embedding_size, 
        return_array=True
    )


def get_cosine_score(
    dataframe: pd.DataFrame, 
    column_name: str, 
    input_text_embed: np.ndarray
) -> float:
    """
    Calculates the cosine similarity between a user query embedding and 
    the embedding stored in a specific row of the DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame row (Pandas Series) 
            containing the embedding.
        column_name (str): The name of the column with the stored embedding.
        input_text_embed (np.ndarray): The user's query embedding.

    Returns:
        float: The cosine similarity score, rounded to two decimal places.

    Explanation for Beginners:
        - Cosine similarity is a measure of how close two vectors are in 
            angle. If they are pointing in the same direction, the score is 1. 
            If they are orthogonal, the score is 0. 
            If they are opposite, it's -1 (though that rarely matters here).
        - Here, we do a simple dot product, but note that for a "true" cosine
            similarity, you'd normally also divide by the norms of the vectors. 
            This code is simplified to demonstrate the process.
    """
    # Dot product is used as a quick measure of similarity here,
    # but a real cosine similarity would also normalize vectors.
    text_cosine_score = round(np.dot(dataframe[column_name], input_text_embed), 2)
    return text_cosine_score


# ----------------------------------------------------------------------------
#       PRINT FUNCTIONS FOR CITATIONS/RESULTS
# ----------------------------------------------------------------------------

def print_text_to_image_citation(
    final_images: Dict[int, Dict[str, Any]],
    print_top: bool = True
) -> None:
    """
    Prints citations for matched images in a color-coded format.

    Args:
        final_images (Dict[int, Dict[str, Any]]): A dictionary containing details 
            about matched images:
            - The index (key) can be 0, 1, 2, ...
            - Each value is another dict with fields like 
                'cosine_score', 'file_name', 'img_path', 'page_num', 
                'page_text', 'image_description', etc.
        print_top (bool, optional): If True, only print the first citation. 
            If False, print them all.

    Explanation for Beginners:
        - Suppose you found 3 images that match the user's query. 
            This dictionary stores all of them.
        - We then loop through and display them nicely with colors.
    """
    color = Color()

    for imageno, image_dict in final_images.items():
        print(
            color.RED + f"Citation {imageno + 1}:"
            f" Matched image path, page number, and page text:\n" + color.END
        )

        print(color.BLUE + "score: " + color.END, image_dict["cosine_score"])
        print(color.BLUE + "file_name: " + color.END, image_dict["file_name"])
        print(color.BLUE + "path: " + color.END, image_dict["img_path"])
        print(color.BLUE + "page number: " + color.END, image_dict["page_num"])
        print(
            color.BLUE + "page text: " + color.END, 
            "\n".join(image_dict["page_text"])
        )
        print(
            color.BLUE + "image description: " + color.END, 
            image_dict["image_description"]
        )

        # If we only want the top result, break after the first.
        if print_top and imageno == 0:
            break


def print_text_to_text_citation(
    final_text: Dict[int, Dict[str, Any]],
    print_top: bool = True,
    chunk_text: bool = True,
) -> None:
    """
    Prints citations for matched text passages.

    Args:
        final_text (Dict[int, Dict[str, Any]]): A dictionary with matched text info.
        print_top (bool, optional): If True, only print the first. Defaults to True.
        chunk_text (bool, optional): If True, print chunked text. 
            If False, print entire page text. Defaults to True.

    Explanation for Beginners:
        - If you do a text search, you might find multiple matches. 
            This dictionary stores the top matches. 
        - We loop through them and print the details (score, file name, etc.).
    """
    color = Color()

    for textno, text_dict in final_text.items():
        print(
            color.RED + f"Citation {textno + 1}: Matched text:\n" + color.END
        )

        print(color.BLUE + "score: " + color.END, text_dict["cosine_score"])
        print(color.BLUE + "file_name: " + color.END, text_dict["file_name"])
        print(color.BLUE + "page_number: " + color.END, text_dict["page_num"])

        # Decide whether to show chunk text or entire page text.
        if chunk_text:
            print(
                color.BLUE + "chunk_number: " + color.END, 
                text_dict["chunk_number"]
            )
            print(
                color.BLUE + "chunk_text: " + color.END, 
                text_dict["chunk_text"]
            )
        else:
            print(
                color.BLUE + "page text: " + color.END, 
                text_dict["text"]
            )

        if print_top and textno == 0:
            break


# ----------------------------------------------------------------------------
#       FUNCTIONS TO FIND SIMILAR IMAGES OR TEXT FROM A QUERY
# ----------------------------------------------------------------------------

def get_similar_image_from_query(
    text_metadata_df: pd.DataFrame,
    image_metadata_df: pd.DataFrame,
    query: str = "",
    image_query_path: str = "",
    column_name: str = "",
    image_emb: bool = True,
    top_n: int = 3,
    embedding_size: int = 128,
) -> Dict[int, Dict[str, Any]]:
    """
    Finds the top-N most similar images from a metadata DataFrame 
    based on either a text query or an image query.

    Args:
        text_metadata_df (pd.DataFrame): DataFrame containing text metadata 
            related to the images (for referencing the page text).
        image_metadata_df (pd.DataFrame): DataFrame containing image metadata 
            (paths, descriptions, embeddings).
        query (str, optional): The text query if searching by text. Defaults to "".
        image_query_path (str, optional): The path to an image if searching by image. 
            Defaults to "".
        column_name (str, optional): The column name in 'image_metadata_df' 
            that contains embeddings. Defaults to "".
        image_emb (bool, optional): If True, use an image as the query. 
            If False, use text as the query. Defaults to True.
        top_n (int, optional): How many best matches to return. Defaults to 3.
        embedding_size (int, optional): Embedding size used for images. Defaults to 128.

    Returns:
        Dict[int, Dict[str, Any]]:
            A dictionary of matches, with each value holding:
            - 'cosine_score'
            - 'image_object'
            - 'file_name'
            - 'img_path'
            - 'page_num'
            - 'page_text'
            - 'image_description'

    Explanation for Beginners:
        - We either embed the user's text or the user's image (depending on 'image_emb').
        - Then we compare that embedding to all the image embeddings in our DataFrame,
            computing a similarity score.
        - We pick the top 'n' highest similarity images.
        - We also retrieve the associated page text so we know the context.
    """
    # Decide which embedding to generate (text or image).
    if image_emb:
        user_query_image_embedding = get_user_query_image_embeddings(
            image_query_path, embedding_size
        )
        # Compute a similarity score for each row in the image metadata
        cosine_scores = image_metadata_df.apply(
            lambda x: get_cosine_score(x, column_name, user_query_image_embedding),
            axis=1,
        )
    else:
        user_query_text_embedding = get_user_query_text_embeddings(query)
        cosine_scores = image_metadata_df.apply(
            lambda x: get_cosine_score(x, column_name, user_query_text_embedding),
            axis=1,
        )

    # Remove any rows that might have a perfect 1.0 match 
    # due to an identical embedding in the dataset.
    cosine_scores = cosine_scores[cosine_scores < 1.0]

    # Find the indices of the top N highest scores
    top_n_indices = cosine_scores.nlargest(top_n).index.tolist()
    top_n_values = cosine_scores.nlargest(top_n).values.tolist()

    final_images: Dict[int, Dict[str, Any]] = {}

    # For each top match, store the details in a dictionary.
    for matched_imageno, idx_value in enumerate(top_n_indices):
        final_images[matched_imageno] = {}

        final_images[matched_imageno]["cosine_score"] = top_n_values[matched_imageno]

        # Load the image from the file path into a Vertex AI Image object.
        final_images[matched_imageno]["image_object"] = Image.load_from_file(
            image_metadata_df.iloc[idx_value]["img_path"]
        )
        final_images[matched_imageno]["file_name"] = image_metadata_df.iloc[idx_value][
            "file_name"
        ]
        final_images[matched_imageno]["img_path"] = image_metadata_df.iloc[idx_value][
            "img_path"
        ]
        final_images[matched_imageno]["page_num"] = image_metadata_df.iloc[idx_value][
            "page_num"
        ]

        # Get the text from the corresponding page in the text metadata.
        page_num = final_images[matched_imageno]["page_num"]
        file_name = final_images[matched_imageno]["file_name"]
        page_text = text_metadata_df[
            (text_metadata_df["page_num"] == page_num)
            & (text_metadata_df["file_name"] == file_name)
        ]["text"].values

        # Avoid duplicates by converting to a NumPy unique array.
        final_images[matched_imageno]["page_text"] = np.unique(page_text)

        # Also store the image description from the DataFrame.
        final_images[matched_imageno]["image_description"] = image_metadata_df.iloc[
            idx_value
        ]["img_desc"]

    return final_images


def get_similar_text_from_query(
    query: str,
    text_metadata_df: pd.DataFrame,
    column_name: str = "",
    top_n: int = 3,
    chunk_text: bool = True,
    print_citation: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Finds the top N most similar text passages from the text metadata 
    DataFrame based on a text query.

    Args:
        query (str): The text query (what the user is searching for).
        text_metadata_df (pd.DataFrame): DataFrame containing text 
            (page text, chunk text, embeddings).
        column_name (str, optional): Which column in the DataFrame has the embeddings. 
            Defaults to "".
        top_n (int, optional): How many best matches to return. Defaults to 3.
        chunk_text (bool, optional): If True, show chunked text in the results. 
            If False, show entire page text. Defaults to True.
        print_citation (bool, optional): Whether to immediately print the results 
            (citations). Defaults to False.

    Returns:
        Dict[int, Dict[str, Any]]:
            A dictionary of top matches, each containing:
            - 'file_name'
            - 'page_num'
            - 'cosine_score'
            - 'chunk_number' and 'chunk_text' if chunk_text is True
            - 'text' if chunk_text is False

    Explanation for Beginners:
        - We embed the user's query text.
        - We compare that embedding to the embeddings in 'text_metadata_df'.
        - We get the highest scores (most similar).
        - We optionally print them right away if 'print_citation' is True.
    """
    # Ensure the column exists
    if column_name not in text_metadata_df.columns:
        raise KeyError(f"Column '{column_name}' not found in 'text_metadata_df'")

    # Embed the user's query text
    query_vector = get_user_query_text_embeddings(query)

    # Calculate similarity (here, just a dot product for demonstration)
    cosine_scores = text_metadata_df.apply(
        lambda row: get_cosine_score(row, column_name, query_vector),
        axis=1,
    )

    # Get top N indices
    top_n_indices = cosine_scores.nlargest(top_n).index.tolist()
    top_n_scores = cosine_scores.nlargest(top_n).values.tolist()

    final_text: Dict[int, Dict[str, Any]] = {}

    # Build a results dictionary
    for matched_textno, index in enumerate(top_n_indices):
        final_text[matched_textno] = {}

        final_text[matched_textno]["file_name"] = text_metadata_df.iloc[index][
            "file_name"
        ]
        final_text[matched_textno]["page_num"] = text_metadata_df.iloc[index][
            "page_num"
        ]
        final_text[matched_textno]["cosine_score"] = top_n_scores[matched_textno]

        # Depending on whether we want chunked text or full page text
        if chunk_text:
            final_text[matched_textno]["chunk_number"] = text_metadata_df.iloc[index][
                "chunk_number"
            ]
            final_text[matched_textno]["chunk_text"] = text_metadata_df["chunk_text"][
                index
            ]
        else:
            final_text[matched_textno]["text"] = text_metadata_df["text"][index]

    # Optionally, print the results as soon as we find them
    if print_citation:
        print_text_to_text_citation(final_text, chunk_text=chunk_text)

    return final_text


# ----------------------------------------------------------------------------
#       FUNCTION TO DISPLAY A SERIES OF IMAGES
# ----------------------------------------------------------------------------

def display_images(
    images: Iterable[Union[str, PIL.Image.Image]], 
    resize_ratio: float = 0.5
) -> None:
    """
    Displays a series of images (provided as either file paths or PIL Image 
    objects) within a Jupyter notebook or IPython environment.

    Args:
        images (Iterable[Union[str, PIL.Image.Image]]): A list or other iterable
            containing either string file paths or PIL Image objects.
        resize_ratio (float, optional): How much to resize the images by. 
            Defaults to 0.5 (50% smaller).

    Returns:
        None (displays images in the notebook).

    Explanation for Beginners:
        - If the item is a file path, we open it as a PIL image first.
        - Then we resize the image to the specified ratio and show it.
        - 'display()' is an IPython function that outputs the image 
            right in the notebook cell.
    """
    pil_images = []

    # Convert any file path strings into PIL Image objects
    for image in images:
        if isinstance(image, str):
            pil_images.append(PIL.Image.open(image))
        else:
            pil_images.append(image)

    # Resize and display each image
    for img in pil_images:
        original_width, original_height = img.size
        new_width = int(original_width * resize_ratio)
        new_height = int(original_height * resize_ratio)
        resized_img = img.resize((new_width, new_height))
        display(resized_img)
        print("\n")  # Just to provide a bit of spacing in the output
