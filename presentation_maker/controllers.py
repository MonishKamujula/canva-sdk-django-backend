"""
Controllers for the presentation_maker app.

Business logic for generating Canva design functions from card content.
"""

import json
import logging
import math
from typing import List, AsyncGenerator

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
def estimate_num_lines(items: List[dict]) -> List[float]:
    """
    Estimate the height in pixels for a list of text content items.
    
    Args:
        items: List of dictionaries, each containing:
            - content: The text content.
            - box_width_px: Width of the box in pixels.
            - font_size_pt: Font size in points.
            - mu: Optional character constant (default 0.56).
            
    Returns:
        List of estimated heights in pixels, corresponding to the input order.
    """
    logger.debug(f"estimate_num_lines batch tool called with {len(items)} items")
    results = []
    
    for item in items:
        content = item.get("content", "")
        box_width_px = item.get("box_width_px", 0)
        font_size_pt = item.get("font_size_pt", 12)
        mu = item.get("mu", 0.56)
        
        # Convert pt to px
        font_size_px = font_size_pt * (96 / 72)

        # Estimate characters per line (CPL)
        # Avoid division by zero
        if mu * font_size_px == 0:
            chars_per_line = 1
        else:
            chars_per_line = box_width_px / (mu * font_size_px)

        # Estimate number of lines
        if chars_per_line <= 0:
             num_lines = 1
        else:
            num_lines = math.ceil(len(content) / chars_per_line)
        
        total_height = (font_size_px * 1.4) * num_lines + (1.1 * font_size_px)
        results.append(total_height)

    return results


@tool
def enforce_design_constraints(items: List[dict]) -> List[dict]:
    """
    Validates and FIXES a list of elements to ensure they fit within the page safe zones.
    
    Args:
        items: List of dictionaries, each containing:
            - left, top, width, height: Element dimensions.
            - page_width, page_height: Page dimensions.
            - padding: Optional safe zone padding (default 40.0).
            
    Returns:
        List of FIXED elements that are guaranteed to be valid.
    """
    fixed_items = []
    
    for item in items:
        # Create a copy to avoid mutating the original dict if needed, 
        # though here we are building a new list.
        # We ensure all necessary keys are present.
        fixed_item = item.copy()
        
        left = float(fixed_item.get("left", 0))
        top = float(fixed_item.get("top", 0))
        width = float(fixed_item.get("width", 0))
        height = float(fixed_item.get("height", 0))
        page_width = fixed_item.get("page_width", 0)
        page_height = fixed_item.get("page_height", 0)
        padding = fixed_item.get("padding", 40.0)
        
        # 1. Enforce Width Constraints
        max_width = page_width - (2 * padding)
        if width > max_width:
            fixed_item["width"] = max_width
            width = max_width
            
        # 2. Enforce Height Constraints
        max_height = page_height - (2 * padding)
        if height > max_height:
            fixed_item["height"] = max_height
            height = max_height

        # 3. Enforce Left Position (Left Wall)
        if left < padding:
            fixed_item["left"] = padding
            left = padding
            
        # 4. Enforce Right Position (Right Wall)
        right_edge = left + width
        if right_edge > (page_width - padding):
            # Shift left to fit
            new_left = page_width - padding - width
            fixed_item["left"] = max(padding, new_left) # Don't push past left wall
            left = fixed_item["left"]

        # 5. Enforce Top Position (Ceiling)
        if top < padding:
            fixed_item["top"] = padding
            top = padding
            
        # 6. Enforce Bottom Position (Floor)
        bottom_edge = top + height
        if bottom_edge > (page_height - padding):
            # Shift up to fit
            new_top = page_height - padding - height
            fixed_item["top"] = max(padding, new_top) # Don't push past ceiling
            
        fixed_items.append(fixed_item)

    return fixed_items

def create_canva_functions(page_dimensions: dict, card: dict) -> str:
    """
    Generate Canva design functions.
    """
    logger.info(f"Generating Canva functions for card: {card.get('title', 'Unknown')}")
    
    all_steps = create_steps(card)
    relevant_canva_doc = handle_rag(all_steps.rag_query)
    steps = ",".join(all_steps.steps)
    
    prompt = f"""
### SYSTEM ROLE
You are an ultra-precise AI Designer for Canva.

### OBJECTIVE
Convert "User Steps" into a single JSON array of Canva functions.

### INPUT CONTEXT
1. **User Steps:** {steps}
2. **Docs:** {relevant_canva_doc}
3. **Dimensions:** {page_dimensions} (W x H)

### MANDATORY TOOL PROTOCOL
You must use the provided tools to ensure mathematical perfection. 

**BATCH PROCESSING REQUIRED:** 
Do NOT call tools one by one. You MUST gather all elements and call the tool ONCE with a list of items.

1. **Step 1: Draft & Measure (BATCH)**
   - Determine `width` and `font_size` for ALL text elements.
   - Call `estimate_num_lines` with a **list** of all text items to get exact `height` for each.

2. **Step 2: Enforce Constraints (BATCH)**
   - Call `enforce_design_constraints` with a **list** of all elements using your drafted `left`, `top`, `width`, `height`.
   - This tool will AUTOMATICALLY FIX any layout issues.

3. **Step 3: Output**
   - The tool output from Step 2 is your FINAL ANSWER.
   - Output that JSON array immediately. Do not modify it.

### DESIGN CONSTRAINTS
- **Safe Zone:** 40px padding on all sides.
- **Colors:** 6-digit hex only (e.g., "#000000").

### OUTPUT FORMAT
- **Single JSON Array:** `[{{...}}, {{...}}]`
- **No Markdown:** Do not use ```json or text blocks.
- **No Whitespace:** Minify the output.
"""
    
    # Use LangChain agent with tool calling
    lang_chain_openai = ChatOpenAI(model="gpt-4o-mini")
    agent = create_react_agent(model=lang_chain_openai, tools=[estimate_num_lines, enforce_design_constraints])
    
    response = agent.invoke({
        "messages": [
            {"role": "user", "content": prompt} 
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


async def stream_canva_functions(page_dimensions: dict, card: dict) -> AsyncGenerator[dict, None]:
    """
    Async generator that streams Canva design elements one by one.
    
    Uses LangGraph's astream_events to capture the final AI message tokens,
    then parses complete JSON objects from the streaming array output.
    
    Yields:
        dict: Individual Canva element definitions (after image replacement)
    """
    logger.info(f"Streaming Canva functions for card: {card.get('title', 'Unknown')}")
    
    # Prepare steps and context (sync operations, run in executor if needed)
    all_steps = create_steps(card)
    relevant_canva_doc = handle_rag(all_steps.rag_query)
    steps = ",".join(all_steps.steps)
    
    prompt = f"""
### SYSTEM ROLE
You are an ultra-precise AI Designer for Canva.

### OBJECTIVE
Convert "User Steps" into a single JSON array of Canva functions.

### INPUT CONTEXT
1. **User Steps:** {steps}
2. **Docs:** {relevant_canva_doc}
3. **Dimensions:** {page_dimensions} (W x H)

### MANDATORY TOOL PROTOCOL
You must use the provided tools to ensure mathematical perfection.

**BATCH PROCESSING REQUIRED:** 
Do NOT call tools one by one. You MUST gather all elements and call the tool ONCE with a list of items.

1. **Step 1: Draft & Measure (BATCH)**
   - Determine `width` and `font_size` for ALL text elements.
   - Call `estimate_num_lines` with a **list** of all text items to get exact `height` for each.

2. **Step 2: Enforce Constraints (BATCH)**
   - Call `enforce_design_constraints` with a **list** of all elements using your drafted `left`, `top`, `width`, `height`.
   - This tool will AUTOMATICALLY FIX any layout issues.

3. **Step 3: Output**
   - The tool output from Step 2 is your FINAL ANSWER.
   - Output that JSON array immediately. Do not modify it.

### DESIGN CONSTRAINTS
- **Safe Zone:** 40px padding on all sides.
- **Colors:** 6-digit hex only (e.g., "#000000").

### OUTPUT FORMAT
- **Single JSON Array:** `[{{...}}, {{...}}]`
- **No Markdown:** Do not use ```json or text blocks.
- **No Whitespace:** Minify the output.
"""
    
    # Use LangChain agent with tool calling
    lang_chain_openai = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    agent = create_react_agent(model=lang_chain_openai, tools=[estimate_num_lines, enforce_design_constraints])
    
    # Buffer to accumulate tokens from the final answer
    buffer = ""
    in_array = False
    brace_depth = 0
    current_object_start = -1
    
    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": prompt}]},
        version="v2"
    ):
        kind = event.get("event")
        
        # We're looking for the final AI message content
        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                token = chunk.content
                buffer += token
                
                # Parse the buffer for complete JSON objects
                i = len(buffer) - len(token)  # Start from where new content was added
                while i < len(buffer):
                    char = buffer[i]
                    
                    # Skip if we're inside a string
                    if char == '"':
                        # Simple string handling - find the closing quote
                        i += 1
                        while i < len(buffer):
                            if buffer[i] == '\\':
                                i += 2  # Skip escaped character
                                continue
                            if buffer[i] == '"':
                                break
                            i += 1
                    elif char == '[' and not in_array:
                        # Start of the array
                        in_array = True
                    elif char == '{':
                        if brace_depth == 0:
                            current_object_start = i
                        brace_depth += 1
                    elif char == '}':
                        brace_depth -= 1
                        if brace_depth == 0 and current_object_start >= 0:
                            # We have a complete object!
                            object_str = buffer[current_object_start:i+1]
                            try:
                                element = json.loads(object_str)
                                # Apply image replacement
                                processed = replace_images([element])
                                if processed:
                                    yield processed[0]
                                    logger.debug(f"Yielded element: {element.get('type', 'unknown')}")
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse element: {e}")
                            current_object_start = -1
                    
                    i += 1
    
    logger.info("Streaming complete")