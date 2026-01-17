"""
Controllers for the presentation_maker app.

Business logic for generating Canva design functions from card content.
"""

import json
import logging
import math
from typing import List

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.ai import use_openai
from core.utils import replace_images
from .rag.canva_rag import handle_rag
from .schemas import StepBreakdown

logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI()


def create_steps(card: dict) -> StepBreakdown:
    """
    Break down card content into design steps.
    
    Args:
        card: Dictionary with 'title' and 'description' keys.
    
    Returns:
        StepBreakdown with steps and RAG queries.
    """
    step_prompt = '''
    # Break down the given card into smaller technical step commands, separated by commas, to generate a design document.
- Input: 
  • card – an object containing:
    – heading: the slide's main title  
    – subtopics: a list of section headings or bullet points  

- Example:  
  If card = {
    "heading": "Add information about animals",
    "subtopics": "animal image", "animal description"
  }  
  then output:  
  Add slide heading "Add information about animals", Add image of an animal centered on the slide, Add paragraph describing the animal at the bottom of the slide.

  Remember that this is a slideshow so if you think you need to expand more on a topic the you can do it. So the output should have more info on subheadings not only the the subtopics.
  
  - If the subheadings have "Images" at the end then be sure to add images.
  - For each step, also list the RAG queries required (choose all that apply):  
  - An element that renders image content and has positional properties.  
  - An element that renders video content and has positional properties.  
  - An element that renders text content and has positional properties. 
  - An element that renders embeds and has positional properties.  
  - An element that renders a table.  
  - An element that renders a vector shape and has positional properties.
    '''
    
    logger.info(f"Creating steps for card: {card['title']}")
    
    response = use_openai(
        step_prompt,
        f"Heading: {card['title']}, Subtopics: {card['description']}",
        "gpt-4o-mini",
        format=StepBreakdown
    )
    return response


@tool
def estimate_pixels(
    content: str,
    box_width_px: float,
    font_size_pt: float,
    mu: float = 0.56,
    debug: bool = False
) -> float:
    """
    Estimate the height in pixels for given text content.

    Args:
        content: The text content.
        box_width_px: Width of the box in pixels.
        font_size_pt: Font size in points.
        mu: Character constant (default 0.56 for Canva Sans / Open Sans).
        debug: If True, prints debug info.

    Returns:
        Estimated height in pixels.
    """
    logger.debug("estimate_pixels tool called")
    
    # Convert pt to px
    font_size_px = font_size_pt * (96 / 72)

    # Estimate characters per line (CPL)
    chars_per_line = box_width_px / (mu * font_size_px)

    # Estimate number of lines
    num_lines = math.ceil(len(content) / chars_per_line)

    if debug:
        logger.debug(f"Font size (px): {font_size_px:.2f}")
        logger.debug(f"Characters per line (CPL): {chars_per_line:.2f}")
        logger.debug(f"Content length: {len(content)}")
        logger.debug(f"Estimated number of lines: {num_lines}")

    total_height = (font_size_px * 1.4) * num_lines + (1.1 * font_size_px)
    return total_height


def create_canva_functions(page_dimensions: dict, card: dict) -> str:
    """
    Generate Canva design functions for a presentation card.
    
    Uses RAG to retrieve relevant Canva SDK documentation and
    LangChain agent for intelligent function generation.
    
    Args:
        page_dimensions: Dictionary with 'width' and 'height' in pixels.
        card: Dictionary with 'title' and 'description' keys.
    
    Returns:
        JSON string of Canva function definitions.
    """
    logger.info(f"Generating Canva functions for card: {card.get('title', 'Unknown')}")
    
    all_steps = create_steps(card)
    relevant_canva_doc = handle_rag(all_steps.rag_query)
    
    prompt = f'''
    You are an AI designer for Canva pages with exceptional aesthetics.
    
    Inputs: The user will give you steps to generate a Canva page, and you will get the relevant Canva documentation to do the task that steps say: {relevant_canva_doc}

    Output: You will generate a single JSON array of Canva functions to generate the page. Example: You need to return this [{{canva functions}}...]. Make Sure to put limited information, to make sure no overlaying elements.

    Tips: Follow the steps, if you think you need to add more info then add it. If you think you will need to remove some then you can do it. But don't overdo it.

    Extra tools:
     - If you are adding a text element, then you can't provide a height for it you don't know the height for it. For find its height use the tool I provided(estimate_num_lines), provide it with is required params. It will give you the height in pixels. Don't worry to add text box close if it is aesthetic.

    - Place every element strictly within the page dimensions.  
    - Output must be a single, compact JSON array: start with [ and end with ], with no comments, extra tags, empty lines, or unnecessary spaces.  
    - Page dimensions (pixels): {page_dimensions}  
    – Top-left = (left=0, top=0)  
    – Top-right = (left={page_dimensions["width"]}, top=0)  
    – Bottom-left = (left=0, top={page_dimensions["height"]})  
    – Bottom-right = (left={page_dimensions["width"]}, top={page_dimensions["height"]})  
    • When a parameter has a restricted set of values (e.g., textAlign), choose only from those allowed options ("start", "center", "end", "justify").
    - If you are adding a text element, then you can't provide a height for it you don't know the height for it. For find its height use the tool I provided(estimate_num_lines), provide it with is required params. It will give you the height in pixels. Don't worry to add text box close if it is aesthetic.
    - Width is a required property for all elements. Always specify a width in pixels.
    - If you need to add any colors then you need to provide a 6-digit hex code not a 3-digit hex code.
    '''

    steps = ",".join(all_steps.steps)
    
    # Use LangChain agent with tool calling
    lang_chain_openai = ChatOpenAI(model="gpt-4o", temperature=0.3)
    agent = create_react_agent(model=lang_chain_openai, tools=[estimate_pixels])
    
    response = agent.invoke({
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": steps}
        ]
    })
    
    # Parse the response
    response_content = response["messages"][-1].content
    logger.info(f"Raw agent response: {response_content}")
    
    # Clean markdown code blocks if present
    if "```" in response_content:
        response_content = response_content.replace("```json", "").replace("```", "").strip()
    
    try:
        parsed_response = json.loads(response_content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse agent response as JSON: {response_content}")
        raise ValueError(f"Agent returned invalid JSON: {e}")
    
    # Replace image references with actual URLs
    functions = replace_images(parsed_response)
    
    return json.dumps(functions, indent=2)