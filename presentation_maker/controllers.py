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
    """
    step_prompt = '''
### TASK
Break down the Input Card into a list of specific, technical design commands.

### INPUT DATA
Heading: {heading}
Subtopics: {subtopics}

### GUIDELINES
1. **Expansion:** You are a presentation expert. If the input is sparse, reasonably expand on the subtopics to fill a slide (e.g., if "Animals" is the topic, add specific steps for "Add an image of a lion" and "Add text describing habitats").
2. **Images:** If subtopics mention "Images" or visual concepts, you MUST include a step to "Add an image of...".
3. **Format:** Output must match the StepBreakdown schema exactly.

### REQUIRED RAG QUERIES
For each step, identify the necessary Canva element types from this list:
- Text element (headings, paragraphs)
- Image element
- Video element
- Embed element
- Table element
- Vector/Shape element
'''
    
    logger.info(f"Creating steps for card: {card['title']}")
    
    response = use_openai(
        f"{step_prompt.format(heading=card['title'], subtopics=card['description'])}",
        "Generate design steps.", # System instruction context
        "gpt-4o-mini",
        format=StepBreakdown
    )
    return response

@tool
def estimate_num_lines(
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
    logger.debug("estimate_num_lines tool called")
    
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


@tool
def check_element_overflow(
    left: float, 
    top: float, 
    width: float, 
    height: float, 
    page_width: float, 
    page_height: float,
    padding: float = 40.0
) -> dict:
    """
    Checks if an element fits within the page safe zones.
    Returns a status and a description of the violation, but NO FIX.
    
    Args:
        left, top, width, height: Element dimensions.
        page_width, page_height: Page dimensions.
        padding: Safe zone padding in pixels (default 40.0).
        
    Returns:
        Dict with status ("OK" or "WARNING") and details.
    """
    
    right_edge = left + width
    bottom_edge = top + height

    # Check for Safe Zone Violations
    violation_left = left < padding
    violation_top = top < padding
    violation_right = right_edge > (page_width - padding)
    violation_bottom = bottom_edge > (page_height - padding)

    is_overflowing = violation_left or violation_top or violation_right or violation_bottom

    if not is_overflowing:
        return {"status": "OK", "message": "Valid."}

    # Construct specific error list for the model to interpret
    errors = []
    if violation_left: 
        errors.append(f"VIOLATION_LEFT: {left} < {padding}")
    if violation_top: 
        errors.append(f"VIOLATION_TOP: {top} < {padding}")
    if violation_right: 
        errors.append(f"VIOLATION_RIGHT: {right_edge} > {page_width - padding}")
    if violation_bottom: 
        errors.append(f"VIOLATION_BOTTOM: {bottom_edge} > {page_height - padding}")

    return {
        "status": "WARNING",
        "message": "Safe Zone Violation Detected.",
        "details": errors  # Model must read this list to know what to fix
    }

def create_canva_functions(page_dimensions: dict, card: dict) -> str:
    """
    Generate Canva design functions using GPT-5 Nano and strict layout validation.
    """
    logger.info(f"Generating Canva functions for card: {card.get('title', 'Unknown')}")
    
    all_steps = create_steps(card)
    relevant_canva_doc = handle_rag(all_steps.rag_query)
    steps = ",".join(all_steps.steps)
    
    prompt = f"""
### SYSTEM ROLE
You are an ultra-precise AI Designer for Canva (GPT-5 Nano). Your outputs must be mathematically perfect JSON arrays. You strictly adhere to design constraints and error correction logic.

### OBJECTIVE
Convert "User Steps" into a single JSON array of Canva functions.

### INPUT CONTEXT
1. **User Steps:** {steps}
2. **Docs:** {relevant_canva_doc}
3. **Dimensions:** {page_dimensions} (W x H)

### MANDATORY TOOL PROTOCOL
You cannot guess dimensions. You cannot guess safety. You must verify every element.

1. **Step 1: Draft & Measure**
   - Determine `width` and `font_size`.
   - Call `estimate_num_lines` to get exact `height` for text.

2. **Step 2: Verify Bounds**
   - Call `check_element_overflow` with your drafted `left`, `top`, `width`, `height`.

3. **Step 3: LOGIC CORRECTION (CRITICAL)**
   - If the tool returns `"status": "WARNING"`, read the `details` list and apply the math below:
     - **VIOLATION_LEFT:** Set `left = 40`.
     - **VIOLATION_TOP:** Set `top = 40`.
     - **VIOLATION_RIGHT:** Set `left = page_width - width - 40`.
     - **VIOLATION_BOTTOM:** Set `top = page_height - height - 40`.
   - **Resize Rule:** If fixing the position is impossible (e.g., width > page_width), you must reduce the element's `width` to fit within the 40px margins.

4. **Step 4: Output**
   - Write the final JSON with the corrected coordinates.

### DESIGN CONSTRAINTS
- **Safe Zone:** 40px padding on all sides. Content must not touch edges.
- **Colors:** 6-digit hex only (e.g., "#000000").
- **Alignment:** Align elements via Left Edge or Center Axis.

### OUTPUT FORMAT
- **Single JSON Array:** `[{{...}}, {{...}}]`
- **No Markdown:** Do not use ```json or text blocks.
- **No Whitespace:** Minify the output.
- **Required Keys:** `width`, `height`, `left`, `top` (in pixels).

### PARAMETER RESTRICTIONS
- **textAlign:** ["start", "center", "end", "justify"] ONLY.
"""
    
    # Use LangChain agent with tool calling
    lang_chain_openai = ChatOpenAI(model="gpt-4o-mini")
    agent = create_react_agent(model=lang_chain_openai, tools=[estimate_num_lines, check_element_overflow])
    
    response = agent.invoke({
        "messages": [
            {"role": "user", "content": prompt} 
            # Note: For Nano/ReAct, it is often better to put the whole context in the first User message 
            # if the System role is reserved for generic behavior, but putting it in System is also valid 
            # depending on your specific model provider's alignment. 
            # Here I put it in User to ensure the Agent sees the Tools instructions as immediate context.
        ]
    })
    
    # Parse the response
    response_content = response["messages"][-1].content
    logger.info(f"Raw agent response: {response_content}")
    
    if "```" in response_content:
        response_content = response_content.replace("```json", "").replace("```", "").strip()
    
    try:
        parsed_response = json.loads(response_content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse agent response as JSON: {response_content}")
        raise ValueError(f"Agent returned invalid JSON: {e}")
    
    functions = replace_images(parsed_response)
    
    return json.dumps(functions, indent=2)