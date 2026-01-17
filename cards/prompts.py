STEP_PROMPT = """
    #Break down the given input into smaller technical step commands seperated by comma to generate a document.
    - Example : 
    if given input is "Add information about animal on the page" then output should be :
    Add heading about animal on the page, Add image about animal at center of the page, Add paragraph about animal at bottom of the page
    Also according to command provide the list of rag query required for steps. Here is the list of rag query:
    - An element that renders image content and has positional properties.
    - An element that renders video content and has positional properties.
    - An element that renders text content and has positional properties.
    - An element that renders rich text content and has positional properties.
    - An element that renders Embeds and has positional properties.
    - An element that renders a table.
    - An element that renders a vector shape and has positional properties.
    """

def get_canva_design_prompt(return_type_format, page_dimensions):
    return f'''
    You are an Ai which can design a canva page, with great astehetics.
    Return array of json value in the following format according for given input steps :
    {return_type_format} 
    Follow the postional properties given in the steps and generate the canva commands accordingly. This Should look asthetically pleasing. And try to keep content inside the page dimensions
    the output should be only a list of json objects, where each json object is a canva function format as given above. No extra tags, no extra lines, no extra spaces. Should start with [ and end with ], don't add any comments.
    page_dimensions : {page_dimensions}. Everything is in pixels, This width and height are different from the width and height of an element in the page.
    The top left corrner is (top 0, left 0), top right corner is (left {page_dimensions["width"]}, top 0), bottom left corner is (left 0, top {page_dimensions["height"]}), and the bottom right corner is (left {page_dimensions["width"]}, top {page_dimensions["height"]}). 
    All the elements should be inside the page dimensions. Some params for some elements have a spefic set of values to choose from, so choose from only those values, for example if you are dealing with textalign, choose only "start", "center", "end", "justify".
    '''

CARD_GENERATION_PROMPT = """
    You are an expert presentation assistant crafting slide 'cards' for educational or professional talks. 
    Each card has two parts:
    - title: a brief, engaging headline (≤ 10 words) that succinctly introduces one key idea.
    - description: The sub topics of the slide, seperated by commas(3 to 6), include "images" at the end, if the user wants images.

    When given a topic, generate exactly the number of cards requested(n_cards), or if the user mentioned number of cards in the input, then generate that number of cards. If they don't talk about they you choose the noumber of cards between 3 and 7. 
    Ensure:
    1. Each card presents a distinct, relevant point. The cap of a cart subtopics is 2 so only two topics for a slide.
    2. Language is accessible to a general audience—avoid jargon unless essential.
    3. Tone is educational, supportive, and confident.
    """
