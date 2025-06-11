# from canva_rag import handle_rag
from openai import OpenAI
from pydantic import BaseModel
from pprint import pprint
import requests
import re
import json
from typing import Dict, List, Optional

# structured outputs
class StepBreakdown(BaseModel):
    steps: List[str]
    rag_query : List[str]

class JsonOutput(BaseModel):
    functions : List[Dict]

class Card(BaseModel):
    title: str
    description: str

class CardList(BaseModel):
    cards: List[Card]

openai_client = OpenAI()

def use_openai(prompt, user_input, model="gpt-4.1", format=None):
    response = openai_client.responses.parse(
        model=model,
        input=[
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": user_input},
    ],
        text_format=format,
    )
    return response

def search_pexels_image(query):
    headers = {"Authorization": "xwjeCY8K2Lz6sYAAwVlUvMEC2rt4cJ2hDVjlEfUdwCTsgyv2jh2MmKQZ"}
    params = {"query": query, "per_page": 1}
    response = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params)
    try: 
        return response.json()['photos'][0]['src']['medium']
    except IndexError as e:
        return "https://i.postimg.cc/jSYRBQWR/image-not-found.png"
    
def replace_images(json_data):
    edited_response = json.loads(json_data)
    for item in edited_response:
        if "ref" in item:
            old = item["ref"]
            item["ref"] = search_pexels_image(old)
    return edited_response
    
def create_steps(user_input):
    step_prompt = f'''
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
    '''
    response = use_openai(step_prompt, user_input, "gpt-4o-mini-2024-07-18", format=StepBreakdown)
    return response.output_parsed


def create_canva_functions(user_input, page_dimensions):
    all_steps = create_steps(user_input)
    return_type_format = handle_rag(all_steps.rag_query)
    prompt = f'''
    You are an Ai which can design a canva page, with great astehetics.
    Return array of json value in the following format according for given input steps :
    {return_type_format} 
    Follow the postional properties given in the steps and generate the canva commands accordingly. This Should look asthetically pleasing. And try to keep content inside the page dimensions
    the output should be only a list of json objects, where each json object is a canva function format as given above. No extra tags, no extra lines, no extra spaces. Should start with [ and end with ], don't add any comments.
    page_dimensions : {page_dimensions}. Everything is in pixels, This width and height are different from the width and height of an element in the page.
    The top left corrner is (top 0, left 0), top right corner is (left {page_dimensions["width"]}, top 0), bottom left corner is (left 0, top {page_dimensions["height"]}), and the bottom right corner is (left {page_dimensions["width"]}, top {page_dimensions["height"]}). 
    All the elements should be inside the page dimensions. Some params for some elements have a spefic set of values to choose from, so choose from only those values, for example if you are dealing with textalign, choose only "start", "center", "end", "justify".
    '''
    steps = ",".join(all_steps.steps)
    response = replace_images(use_openai(prompt, steps, model="gpt-4.1-2025-04-14", format=JsonOutput).output_parsed.functions)
    return json.dumps(response, indent=2)


def create_cards_from_user_input(user_input: str, n_cards: Optional[int] = None) -> CardList:
    """
    Generate presentation cards based on free-form user input.

    Args:
        user_input (str): Free-form text from which AI should extract the topic.
        n_cards (Optional[int]): Desired number of cards. If None, the AI chooses (3–8).

    Returns:
        CardList: Parsed Pydantic model containing the generated cards.
    """
    # Base prompt describing the card structure and tone
    prompt = """
    You are an expert presentation assistant crafting slide 'cards' for educational or professional talks. 
    Each card has two parts:
    - title: a brief, engaging headline (≤ 10 words) that succinctly introduces one key idea.
    - description: The sub topics of the slide, seperated by commas(3 to 6), include "images" at the end, if the user wants images.

    When given a topic, generate exactly the number of cards requested(n_cards), or if the user mentioned number of cards in the input, then generate that number of cards. If they don't talk about they you choose the noumber of cards between 3 and 7. 
    Ensure:
    1. Each card presents a distinct, relevant point.
    2. Language is accessible to a general audience—avoid jargon unless essential.
    3. Tone is educational, supportive, and confident.
    """

    # Build the instruction for the AI, depending on whether n_cards was provided
    if n_cards >= 1:
        instruction = (
            f"Extract the main topic from the following input:\n\n\"{user_input}\"\n\n"
            f"Then generate exactly {n_cards} unique presentation cards about that topic."
        )
    else:
        instruction = (
            f"Extract the main topic from the following input:\n\n\"{user_input}\"\n\n"
        )

    # Call the OpenAI wrapper, instructing it to parse into CardList
    response = use_openai(
        prompt,
        instruction,
        model="gpt-4o-mini-2024-07-18",
        format=CardList
    )

    all_cards = response.output_parsed.cards

    serialized_cards = []
    for card in all_cards:
        serialized_card = {
            "title": card.title,
            "description": card.description,
        }
        serialized_cards.append(serialized_card)


    return serialized_cards