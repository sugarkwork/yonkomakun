import os
import sys
import json
import pickle
import hashlib
import random

from dotenv import load_dotenv
from ultralytics import YOLO
from openai import OpenAI
from fastapi import FastAPI, Request, Response, status
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageFont, ImageChops
import requests
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import MeCab


load_dotenv()


app = FastAPI()
templates = Jinja2Templates(directory="templates")


def save_memory(key, val):
    pickle_file = 'memory.pkl'
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            memory = pickle.load(f)
    else:
        memory = {}
    memory[key] = val
    with open(pickle_file, 'wb') as f:
        pickle.dump(memory, f)


def load_memory(key, defval=None):
    pickle_file = 'memory.pkl'
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            memory = pickle.load(f)
    else:
        memory = {}
    return memory.get(key, defval)


model = YOLO('yolov8x.pt')


client = OpenAI(api_key=os.getenv('API_KEY'))


@app.post("/story/")
def base_story(theme="") -> dict:
    cache = load_memory(f'cache_theme_{theme}', {})
    if cache:
        return cache

    if theme:
        theme = f' The theme is "{theme}".'
    
    with open('prompts/base_story.json', 'r') as file:
        base_prompt = file.read()

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "user", "content": f"Please create a four-panel comic that is lightly humorous.{theme}"},
        ],
        tools=eval(base_prompt),
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    tool_calls = message.tool_calls

    tool_call = tool_calls[0]
    data = json.loads(tool_call.function.arguments)

    save_memory(f'cache_theme_{theme}', data)

    return data


@app.post("/manga/")
def create_manga(theme="", request: Request=None):
    return templates.TemplateResponse("generate.html", {
        "theme": theme,
        "request": request
    })


class Dialogue:
    def __init__(self, dialogue: dict):
        self.character = dialogue.get('character', '')
        self.speech = dialogue.get('speech', '')
        self.inner_thoughts = dialogue.get('inner_thoughts', '')


class Panel:
    def __init__(self, panel: dict):
        self.setting = panel.get('setting', '')
        self.actions = panel.get('actions', '')
        self.state = panel.get('state', '')
        self.drawing_prompt = panel.get('drawing_prompt', '')

        self.dialogue = []
        for dialogue in panel.get('dialogue', []):
            self.dialogue.append(Dialogue(dialogue))

        self.image = None
    
    def get_image_prompt(self):
        return f"{self.setting}\n{self.actions}\n{self.state}\n{self.drawing_prompt}"
    
    def generate_image(self) -> Image:
        self.image = generate_image(self.get_image_prompt())
        return self.image


def get_panels(prompt: dict) -> list[Panel]:
    panels = []
    for panel in prompt.get('panels', []):
        pane = Panel(panel)
        pane.generate_image()
        panels.append(pane)

    return panels


@app.exception_handler(RequestValidationError)
async def handler(request:Request, exc:RequestValidationError):
    print(exc)
    return JSONResponse(content={}, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


class StoryPanelData(BaseModel):
    story: str
    number: int = 0


def wakati(text: str):
    # MeCabで分かち書き
    tagger = MeCab.Tagger('-Owakati')
    return tagger.parse(text.replace('。','\n').replace('、','\n')).split()


def draw_vertical_text(text, font_path, image_size, font_size, rect):
    # 縦書き画像を作成
    image = Image.new('RGBA', image_size, color=(255,255,255,0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)

    height_len = (rect[3] - rect[1]) / font_size  # 縦書きの文字数
    line_len = 0

    x, y = (rect[2] - font_size), rect[1]

    for word in text:
        next_len = len(word)
        if line_len + next_len >= height_len:
            y = rect[1]
            x -= font_size
            line_len = 0

        for char in word:
            draw.text((x, y), char, font=font, fill=(0,0,0))
            y += font_size
            line_len += 1

    return image


def generate_fukidasi(text, height=180, round=True, border=4):
    # ふきだし画像を作成
    wakati_text = wakati(text)
    print(wakati_text)

    image_size = (1024, 1024)
    rect = (10, 10, 1024, height)

    image = draw_vertical_text(wakati_text, font_path, image_size, font_size, rect)
    gray_image = image.convert("L")
    inverted_image = ImageChops.invert(gray_image)
    bbox = inverted_image.getbbox()
    cropped_image = image.crop(bbox) # ふきだしの余白を削除

    if round:
        # ふきだしの余白を削除した画像を円形にする
        new_width = max(int(cropped_image.width * 1.6), font_size * 4)
        new_height = int(cropped_image.height * 1.6)
        new_image = Image.new("RGBA", (new_width, new_height), color=(255,255,255,0))
        paste_position = ((new_width - cropped_image.width) // 2, (new_height - cropped_image.height) // 2)
        draw = ImageDraw.Draw(new_image)
        ellipse_bbox = [border, border, new_width -border, new_height -border]
        draw.ellipse(ellipse_bbox, fill="white", outline="black", width=border)
        alpha_channel = cropped_image.getchannel('A')
        new_image.paste(cropped_image, paste_position, mask=alpha_channel)
    else:
        new_width = max(int(cropped_image.width * 1.2), font_size * 4)
        new_height = int(cropped_image.height * 1.2)
        new_image = Image.new("RGBA", (new_width, new_height), color=(255,255,255,0))
        paste_position = ((new_width - cropped_image.width) // 2, (new_height - cropped_image.height) // 2)
        draw = ImageDraw.Draw(new_image)
        rectangle_bbox = [border, border, new_width -border, new_height -border]
        draw.rectangle(rectangle_bbox, fill="white", outline="black", width=border)
        alpha_channel = cropped_image.getchannel('A')
        new_image.paste(cropped_image, paste_position, mask=alpha_channel)

    return new_image


def is_fully_contained(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2

    return (x3 <= x1 < x2 <= x4) and (y3 <= y1 < y2 <= y4)


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    if x2 < x3 or x4 < x1:
        return False
    if y2 < y3 or y4 < y1:
        return False
    return True


def overlapping_area(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2

    overlapping_x1 = max(x1, x3)
    overlapping_y1 = max(y1, y3)

    overlapping_x2 = min(x2, x4)
    overlapping_y2 = min(y2, y4)

    overlapping_width = max(0, overlapping_x2 - overlapping_x1)
    overlapping_height = max(0, overlapping_y2 - overlapping_y1)

    return overlapping_width * overlapping_height


animals_and_people = ["person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "teddy bear"]


font_path = 'fonts/GenEiAntique/GenEiAntiqueNv5-M.ttf'
font_size = 30


@app.post("/panel/")
def generate_panel(story_panel: StoryPanelData) -> dict:
    panels = get_panels(json.loads(story_panel.story))
    panel = panels[story_panel.number]
    panel_image = panel.generate_image()
    results = model(panel_image)

    names = results[0].names

    avoid_rects = []
    
    for r in results:
        clss = r.boxes.cls.cpu().int().tolist()
        boxes = r.boxes.xyxy.cpu().tolist()
        for box, cls in zip(boxes, clss):
            label = str(names[cls])
            if label not in animals_and_people:
                continue
            avoid_rects.append(box)

    print("avoid_rects = ", avoid_rects)

    def draw_message(text, round=True):
        image = generate_fukidasi(text, round=round, height=random.randint(180, 250))
        image.save(get_image_path(panel.get_image_prompt(), "_selif"))
        alpha_channel = image.getchannel('A')

        margin = 50

        min_overlap_size = sys.maxsize
        min_overlap_rect = None

        panel_image_box = (0, 0, panel_image.width, panel_image.height)

        count_x = int((panel_image.width - image.width) / 10)
        count_y = int((panel_image.height) / 10)

        for dummy_x in range(count_x):
            x = panel_image.width - image.width - (dummy_x * 10) - margin
            for dummy_y in range(count_y):
                y = dummy_y * 10 + margin

                image_box = (x, y, x + image.width, y + image.height)
                if not is_fully_contained(image_box, panel_image_box):
                    continue

                overlapping = False
                for avoid_rect in avoid_rects:
                    if is_overlapping(image_box, avoid_rect):
                        overlapping = True
                        overlap_size = overlapping_area(image_box, avoid_rect)
                        if overlap_size < min_overlap_size:
                            min_overlap_size = overlap_size
                            min_overlap_rect = image_box
                        break

                if not overlapping:
                    panel_image.paste(image, (x, y), mask=alpha_channel)
                    avoid_rects.append(image_box)
                    return

        panel_image.paste(image, min_overlap_rect, mask=alpha_channel)

    for dialogue in panel.dialogue:
        if len(dialogue.speech) > 0:
            draw_message(dialogue.speech, round=True)
        if len(dialogue.inner_thoughts) > 0:
            draw_message(dialogue.inner_thoughts, round=False)

    panel_image.save(get_image_path(panel.get_image_prompt(), "_selif"))


    return {'path': f'/panel_image/?name={get_image_hash(panel.get_image_prompt())}_selif.png' }


def get_image_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def get_image_path(text, add_info=""):
    return f'image_cache/{get_image_hash(text)}{add_info}.png'


@app.get("/panel_image/")
def panel_image(name: str):
    print(name)
    target_name = f'image_cache/{name}'
    if not os.path.exists(target_name):
        print(f'Image not found: {target_name}')
        with open('images/notfound.png', "rb") as f:
            return Response(content=f.read(), media_type="image/png")
    with open(target_name, "rb") as f:
        return Response(content=f.read(), media_type="image/png")


@app.post("/image/")
def generate_image(text):
    cache_name = get_image_path(text)

    if not os.path.exists('image_cache'):
        os.mkdir('image_cache')

    if not os.path.exists(cache_name):
        response = client.images.generate(
            model="dall-e-3",
            prompt=text,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url

        response = requests.get(image_url)
        response.raise_for_status()

        with open(cache_name, "wb") as f:
            f.write(response.content)

    image = Image.open(cache_name)
    return image


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request
    })


@app.get("/generating.png")
def generating_image():
    with open("images/generating.png", "rb") as f:
        return Response(content=f.read(), media_type="image/png")
    

@app.get("/favicon.ico")
def favicon_image():
    with open("images/favicon.ico", "rb") as f:
        return Response(content=f.read(), media_type="image/x-icon")


# uvicorn main:app --reload --port 7861
