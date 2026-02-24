"""
Controllers for the presentation_maker app.

Business logic for generating Canva design functions from card content.
"""

import json
import logging
import math
from typing import List, AsyncGenerator

from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.config import get_stream_writer

from core.ai import use_openai
from core.utils import replace_images
from .rag.canva_rag import handle_rag
from .schemas import StepBreakdown

logger = logging.getLogger(__name__)



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
def get_estimate_height(items: List[dict] = []) -> List[float]:
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
    writer = get_stream_writer()
    logger.debug(f"get_estimate_height batch tool called with {len(items)} items")
    
    writer({
        "type": "agent_thinking",
        "stage": "get_estimate_heights",
        "message": f"Calculating text heights for {len(items)} elements..."
    })
    
    results = []
    
    for i, item in enumerate(items):
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
        
        writer({
            "type": "agent_thinking",
            "stage": "get_estimate_heights",
            "message": f"Element {i+1}/{len(items)}: estimated height = {total_height:.0f}px"
        })

    writer({
        "type": "agent_thinking",
        "stage": "get_estimate_heights",
        "message": "Height estimation complete."
    })
    
    return results


@tool
def enforce_design_constraints(items: List[dict], page_width: int, page_height: int, padding: float = 40.0) -> List[dict]:
    """
    Validates and FIXES a list of elements to ensure they fit within the page safe zones.
    
    Args:
        items: List of dictionaries. EACH item must be the FULL element object (including type, content, etc.) plus 'left', 'top', 'width', 'height'.
        page_width: Width of the page in pixels.
        page_height: Height of the page in pixels.
        padding: Optional safe zone padding (default 40.0).
            
    Returns:
        List of FIXED elements that are guaranteed to be valid.
    """
    writer = get_stream_writer()
    
    writer({
        "type": "agent_thinking",
        "stage": "enforce_constraints",
        "message": f"Validating {len(items)} elements against page bounds..."
    })
    
    fixed_items = []
    
    for i, item in enumerate(items):
        # Create a copy to avoid mutating the original dict
        fixed_item = item.copy()
        
        left = float(fixed_item.get("left", 0))
        top = float(fixed_item.get("top", 0))
        width = float(fixed_item.get("width", 0))
        height = float(fixed_item.get("height", 0))
        # page_width and page_height are now passed as args, not in item
        
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
        
        # Stream each fixed element as an update (same index as preview)
        processed = replace_images([fixed_item])
        writer({
            "type": "element_update",
            "index": i,
            "data": processed[0] if processed else fixed_item
        })
    
    writer({
        "type": "agent_thinking",
        "stage": "enforce_constraints",
        "message": "All elements validated and fixed."
    })

    return json.dumps(fixed_items)


async def stream_canva_functions(page_dimensions: dict, card: dict) -> AsyncGenerator[dict, None]:
    """
    Two-phase async generator that streams Canva design elements.
    
    Phase 1 (element_preview): Parses complete Canva elements from LLM
    tool_call_chunks as the LLM generates enforce_design_constraints args.
    Each element is a fully renderable Canva element sent immediately.
    
    Phase 2 (element_update): After enforce_design_constraints executes,
    each fixed element is emitted via get_stream_writer() with the same
    index, so the frontend can update in-place.
    
    Also streams agent_thinking events from tools for UI status.
    
    Yields:
        dict: Event dicts with 'type' key: element_preview, element_update,
              agent_thinking
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
4. **Schema Rules:**
   - Text elements: MUST use `children: ["Your text"]`. Do NOT use "content" or "text".
   - Image elements: MUST use `ref`.

### MANDATORY TOOL PROTOCOL
You must use the provided tools to ensure mathematical perfection.

**BATCH PROCESSING REQUIRED:** 
1. **Step 1: Estimate Text Height**
   - Gather all TEXT elements into a list.
   - Call `get_estimate_height(items=[{{"content": "...", "box_width_px": 123, "font_size_pt": 12}}, ...])`.
   - USE the returned heights for your elements.

2. **Step 2: Enforce Constraints**
   - Gather ALL elements (text, images, shapes) into a list.
   - EACH item in the list MUST be the COMPLETE element object (type, content, styling, etc.) combined with your proposed `left`, `top`, `width`, `height`.
   - Call `enforce_design_constraints` with a SINGLE dictionary argument:
     {{
       "items": [ ... your list of elements ... ],
       "page_width": {page_dimensions['width']},
       "page_height": {page_dimensions['height']}
     }}
   - This tool will return the FIXED list of elements.

3. **Step 3: Final Output**
   - The JSON array returned by `enforce_design_constraints` is your FINAL ANSWER.
   - Output it immediately as a single JSON array.

### DESIGN CONSTRAINTS
- **Safe Zone:** 40px padding on all sides.
- **Colors:** 6-digit hex only (e.g., "#000000").

### OUTPUT FORMAT
- **Single JSON Array:** `[{{...}}, {{...}}]`
- **No Markdown:** Do not use ```json or text blocks.
- **No Whitespace:** Minify the output.
"""
    
    # Use LangChain 1 agent with tool calling
    agent = create_agent(model="openai:gpt-4o-mini", tools=[get_estimate_height, enforce_design_constraints])
    
    # --- Phase 1 state: parse preview elements from tool_call_chunks ---
    args_buffer = ""
    brace_depth = 0
    current_object_start = -1
    in_string = False
    escape_next = False
    preview_index = 0
    current_tool_name = None  # Track which tool is being called
    
    async for stream_mode, chunk in agent.astream(
        {"messages": [{"role": "user", "content": prompt}]},
        stream_mode=["messages", "custom"],
        config={"recursion_limit": 100}
    ):
        # ─── CUSTOM events (Phase 2 + agent_thinking) ───
        if stream_mode == "custom":
            # These come from get_stream_writer() inside tools
            # Already structured as {type, index, data, ...}
            yield chunk
        
        # ─── MESSAGES events (Phase 1 — preview from tool call args) ───
        elif stream_mode == "messages":
            token, metadata = chunk
            
            # Check for tool_call_chunks (LLM building tool call arguments)
            if hasattr(token, "tool_call_chunks") and token.tool_call_chunks:
                for tc in token.tool_call_chunks:
                    # Detect which tool is being called from the first chunk
                    tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    if tc_name:
                        # New tool call starting — reset parser state
                        current_tool_name = tc_name
                        args_buffer = ""
                        brace_depth = 0
                        current_object_start = -1
                        in_string = False
                        escape_next = False
                        logger.debug(f"Tool call detected: {current_tool_name}")
                    
                    # Only parse element previews from enforce_design_constraints
                    if current_tool_name != "enforce_design_constraints":
                        continue
                    
                    args_fragment = tc.get("args", "") if isinstance(tc, dict) else getattr(tc, "args", "")
                    if not args_fragment:
                        continue
                    
                    # Accumulate tool call args and parse complete element objects
                    for char in args_fragment:
                        args_buffer += char
                        
                        # Handle string literals (skip brace counting inside strings)
                        if escape_next:
                            escape_next = False
                            continue
                        if char == '\\'  and in_string:
                            escape_next = True
                            continue
                        if char == '"':
                            in_string = not in_string
                            continue
                        if in_string:
                            continue
                        
                        # Track brace depth to find complete JSON objects
                        if char == '{':
                            if brace_depth == 0:
                                current_object_start = len(args_buffer) - 1
                            brace_depth += 1
                        elif char == '}':
                            brace_depth -= 1
                            if brace_depth == 0 and current_object_start >= 0:
                                # Complete element object found!
                                object_str = args_buffer[current_object_start:len(args_buffer)]
                                try:
                                    element = json.loads(object_str)
                                    # Apply image replacement and yield as preview
                                    processed = replace_images([element])
                                    yield {
                                        "type": "element_preview",
                                        "index": preview_index,
                                        "data": processed[0] if processed else element
                                    }
                                    logger.debug(f"Yielded preview element {preview_index}: {element.get('type', 'unknown')}")
                                    preview_index += 1
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Failed to parse preview element: {e}")
                                current_object_start = -1
    
    logger.info(f"Streaming complete. {preview_index} preview elements sent.")