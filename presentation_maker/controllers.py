from .rag.canva_rag import handle_rag
from openai import OpenAI
from pydantic import BaseModel
from pprint import pprint
import requests
import re
import json
from typing import Dict, List, Any

# structured outputs
class StepBreakdown(BaseModel):
    steps: List[str]
    rag_query : List[str]

class ListOfDicts(BaseModel):
    functions: List[Dict[str, Any]]

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
    print("Json_data: ", type(json_data))
    edited_response = json_data
    for item in edited_response:
        if "ref" in item:
            old = item["ref"]
            item["ref"] = search_pexels_image(old)
    return edited_response
    
def create_steps(card):
    step_prompt = '''
    # Break down the given card into smaller technical step commands, separated by commas, to generate a design document.
- Input: 
  • card – an object containing:
    – heading: the slide’s main title  
    – subtopics: a list of section headings or bullet points  

- Example:  
  If card = {
    "heading": "Add information about animals",
    "subtopics": "animal image", "animal description"
  }  
  then output:  
  Add slide heading “Add information about animals”, Add image of an animal centered on the slide, Add paragraph describing the animal at the bottom of the slide.

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
    print("________"*10)
    print(card["title"], card["description"])
    print("________"*10)
    response = use_openai(step_prompt, f"Heading: {card["title"]}, Subtopics: {card["description"]}", "gpt-4o-mini-2024-07-18", format=StepBreakdown)
    return response.output_parsed


def create_canva_functions(page_dimensions, card):
    print("________"*10)
    all_steps = create_steps(card)
    relevent_canva_doc = handle_rag(all_steps.rag_query)
    prompt = f'''
    You are an AI designer for Canva pages with exceptional aesthetics. 
    
    Inputs: The user will give you steps to generate a Canva page, and you will get the relevant Canva documentation to do the task that steps say: {relevent_canva_doc}

    Output: You will generate a single JSON array of Canva functions to generate the page. Exaple: You need to return this [{{canva functions}}...]. Make Sure to put limited information, to make sure no overlaying elements.

    Tips: Follow the steps, if you think you need to add more info then add it. If you think you will need to remove some then you can do it. But don't overdo it.

    - Place every element strictly within the page dimensions.  
    - Output must be a single, compact JSON array: start with [ and end with ], with no comments, extra tags, empty lines, or unnecessary spaces.  
    - Page dimensions (pixels): {page_dimensions}  
    – Top-left = (left=0, top=0)  
    – Top-right = (left={page_dimensions["width"]}, top=0)  
    – Bottom-left = (left=0, top={page_dimensions["height"]})  
    – Bottom-right = (left={page_dimensions["width"]}, top={page_dimensions["height"]})  
    • When a parameter has a restricted set of values (e.g., textAlign), choose only from those allowed options (“start”, “center”, “end”, “justify”).
    '''

    steps = ",".join(all_steps.steps)


    response = openai_client.responses.create(
        model="gpt-4.1-2025-04-14",
        input=[
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": steps},
    ]
    )

    response = json.loads(str(response.output_text))

    print(type(response))

    print(type(response), response)

    functions = replace_images(response)


    return json.dumps(functions, indent=2)