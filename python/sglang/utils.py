"""Common utilities."""

import base64
import importlib
import json
import logging
import signal
import sys
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from json import dumps
from typing import Union,List

import numpy as np
import requests
import pandas as pd
from .global_config import global_config

logger = logging.getLogger(__name__)


def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


def is_same_type(values: list):
    """Return whether the elements in values are of the same type."""
    if len(values) <= 1:
        return True
    else:
        t = type(values[0])
        return all(isinstance(v, t) for v in values[1:])


def read_jsonl(filename: str):
    """Read a JSONL file."""
    rets = []
    with open(filename) as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            rets.append(json.loads(line))
    return rets


def dump_state_text(filename: str, states: list, mode: str = "w"):
    """Dump program state in a text file."""
    from sglang.lang.interpreter import ProgramState

    with open(filename, mode) as fout:
        for i, s in enumerate(states):
            if isinstance(s, str):
                pass
            elif isinstance(s, ProgramState):
                s = s.text()
            else:
                s = str(s)

            fout.write(
                "=" * 40 + f" {i} " + "=" * 40 + "\n" + s + "\n" + "=" * 80 + "\n\n"
            )


class HttpResponse:
    def __init__(self, resp):
        self.resp = resp

    def json(self):
        return json.loads(self.resp.read())

    @property
    def status_code(self):
        return self.resp.status


def http_request(url, json=None, stream=False, api_key=None, verify=None):
    """A faster version of requests.post with low-level urllib API."""
    headers = {"Content-Type": "application/json; charset=utf-8"}

    # add the Authorization header if an api key is provided
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    if stream:
        return requests.post(url, json=json, stream=True, headers=headers)
    else:
        req = urllib.request.Request(url, headers=headers)
        if json is None:
            data = None
        else:
            data = bytes(dumps(json), encoding="utf-8")

        try:
            resp = urllib.request.urlopen(req, data=data, cafile=verify)
            return HttpResponse(resp)
        except urllib.error.HTTPError as e:
            return HttpResponse(e)


def encode_image_base64(image_path: Union[str, bytes]):
    """Encode an image in base64."""
    if isinstance(image_path, str):
        with open(image_path, "rb") as image_file:
            data = image_file.read()
            return base64.b64encode(data).decode("utf-8")
    elif isinstance(image_path, bytes):
        return base64.b64encode(image_path).decode("utf-8")
    else:
        # image_path is PIL.WebPImagePlugin.WebPImageFile
        image = image_path
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


def encode_frame(frame):
    import cv2  # pip install opencv-python-headless
    from PIL import Image

    # Convert the frame to RGB (OpenCV uses BGR by default)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to PIL Image to easily convert to bytes
    im_pil = Image.fromarray(frame)

    # Convert to bytes
    buffered = BytesIO()

    # frame_format = str(os.getenv('FRAME_FORMAT', "JPEG"))

    im_pil.save(buffered, format="PNG")

    frame_bytes = buffered.getvalue()

    # Return the bytes of the frame
    return frame_bytes


def encode_video_base64(video_path: str, num_frames: int = 16):
    import cv2  # pip install opencv-python-headless

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file:{video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"target_frames: {num_frames}")

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            # Handle the case where the frame could not be read
            # print(f"Warning: Could not read frame at index {i}.")
            pass

    cap.release()

    # Safely select frames based on frame_indices, avoiding IndexError
    frames = [frames[i] for i in frame_indices if i < len(frames)]

    # If there are not enough frames, duplicate the last frame until we reach the target
    while len(frames) < num_frames:
        frames.append(frames[-1])

    # Use ThreadPoolExecutor to process and encode frames in parallel
    with ThreadPoolExecutor() as executor:
        encoded_frames = list(executor.map(encode_frame, frames))

    # encoded_frames = list(map(encode_frame, frames))

    # Concatenate all frames bytes
    video_bytes = b"".join(encoded_frames)

    # Encode the concatenated bytes to base64
    video_base64 = "video:" + base64.b64encode(video_bytes).decode("utf-8")

    return video_base64


def _is_chinese_char(cp: int):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def find_printable_text(text: str):
    """Returns the longest printable substring of text that contains only entire words."""
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # After the symbol for a new line, we flush the cache.
    if text.endswith("\n"):
        return text
    # If the last token is a CJK character, we print the characters.
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
        return text[: text.rfind(" ") + 1]


def graceful_registry(sub_module_name: str):
    def graceful_shutdown(signum, frame):
        logger.info(
            f"{sub_module_name} Received signal to shutdown. Performing graceful shutdown..."
        )
        if signum == signal.SIGTERM:
            logger.info(f"{sub_module_name} recive sigterm")

    signal.signal(signal.SIGTERM, graceful_shutdown)


class LazyImport:
    """Lazy import to make `import sglang` run faster."""

    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
        self._module = None

    def _load(self):
        if self._module is None:
            module = importlib.import_module(self.module_name)
            self._module = getattr(module, self.class_name)
        return self._module

    def __getattr__(self, name: str):
        module = self._load()
        return getattr(module, name)

    def __call__(self, *args, **kwargs):
        module = self._load()
        return module(*args, **kwargs)


def _dataframe_info_simple(df:pd.DataFrame, df_name:str, comments=None):
    """
    根据 dataframe 获取 dataframe description 信息
    :param df: 输入 dataframe
    :param df_name: dataframe name
    :param comments: 列名的备注信息, dict
    :return: 能反馈 dataframe 的信息
    """

    # df_info_template_simple = """/*\nDetails about the '{df_name}' dataframe include the data types, comments, the column values info as follows:\n{desc_info}\n*/"""
    df_info_template_simple = """/*\nDetails about the '{df_name}' dataframe that can be used as follows:\n{desc_info}\n*/"""
    # df_info_template_simple = """/*\n'{df_name}' each column information:\n {desc_info}\n*/"""
    info_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Contains NaN": df.isnull().any(),
        "Is Unique": df.nunique() == len(df)
    }).reset_index(drop=True)

    # 添加 Example Values 列，使用所有唯一值但最多取三个



    if comments is not None:
        # 将comments转换为一个字典，以便快速查找
        comments_dict = {item["content"]: {"comment": item["comment"], "info": item["info"]} for item in comments}
        # 为每一列添加comment和info信息
        comment_value = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("comment", ""))
        info_df.insert(4, "Comment", comment_value)

        # info_df['Comment'] = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("comment", ""))
        # info_df['Info'] = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("info", ""))
    
    info_df_new = info_df.set_index('Column Name', drop=True).transpose()
    desc_info_dict = info_df_new.to_dict()

    # desc_info_lines = [f"- '{key}': {value}" for key, value in desc_info_dict.items()]
    # desc_info_lines = [f"- '{key}': {value}" for key, value in desc_info_dict.items()]
    desc_info_lines = []
    for key, value in desc_info_dict.items():
        comment = value.get("Comment", "")
        if comment:
            comment_str = "means " + comment + "."
        else:
            comment_str = ""

        data_type = value["Data Type"]
        
        contains_nan = value["Contains NaN"]
        if contains_nan:
            contains_nan_str = "contains NaN, "
        else:
            contains_nan_str = ""
        
        is_unique = value["Is Unique"]
        if is_unique:
            unique_str = "is unique, "
        else:
            unique_str = ""
            # unique_str = "is not unique, "
        

        if ("float" in data_type) or ("int" in data_type):
            unique_str = ""

        dil = f"- '{key}' {data_type}, {unique_str}{contains_nan_str}{comment_str} Example Values: {global_config.table_insert_sep_token +  global_config.table_insert_embed_token + global_config.table_insert_sep_token}"
        desc_info_lines.append(dil)

    desc_info = "\n".join(desc_info_lines)

    desc_info = desc_info.replace(", '...']", ", ...]")

    df_info = df_info_template_simple.format(
        df_name=df_name,
        desc_info=desc_info,
    )
    
    return df_info

# NOTE: build question and return dfs
def build_table_question(table_urls:List[str], query:str):
    """
    Build the instruction text for the user question.

    Args:
        conv (dict): A dictionary containing conversation information. It should contain the following keys: csv_abs_paths, df_names, query.

    Returns:
        str: The generated question string.

    """
    
    pref = '''With several pandas dataframes available, your task is to write the Python code to address the user's question.\n\n## Follow this format:\nQuestion: The user's query.\nThought: Evaluate the dataframes and the question to determine the solution.\nPython code: Generate the Python code, within ```python ... ```.\n\n## Details about the dataframes:\n\n'''
    
    df_lst = []
    for path in table_urls:
        df = pd.read_csv(
            path,
            encoding="utf-8",
            low_memory=False,
            nrows=global_config.table_read_nrows
        )
        
        # FIXME: temp solution
        filename = path.rsplit("/",maxsplit=1)[-1]
        df_name = filename.split(".", maxsplit=1)[0]
        df._nick_name = df_name
        df_lst.append(df)

    df_info_list = [_dataframe_info_simple(df,df._nick_name) for df in df_lst]
    suf = '''\n\nQuestion: ''' + query + '\n'
    return pref + '\n\n'.join(df_info_list) + suf, df_lst

