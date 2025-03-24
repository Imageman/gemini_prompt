import os
import json
import traceback
import time
from loguru import logger
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm
import sys

# Constants
API_KEY_ENV_VAR = "GOOGLE_API_KEY"
PROMPT_FILE = "./prompt.txt"
OUTPUT_FILE = "prompts_meat.txt"
RAW_RESPONSES_FILE = "raw_responses.txt" # Файл для сохранения исходных ответов
GENERATION_COUNT = 10
MODEL_NAME = "models/gemini-2.0-flash-exp" # Используем flash model
INITIAL_DELAY = 3 # Начальная задержка между запросами


def load_api_key() -> str:
    """Loads the Google Gemini API key from the .env file."""
    load_dotenv()
    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        raise ValueError(f"API key not found in environment variable: {API_KEY_ENV_VAR}")
    return api_key


def read_prompt_from_file(file_path: str) -> str:
    """Reads the prompt from the specified file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading prompt file: {file_path}, {e}")


def generate_responses(api_key: str, prompt: str, generation_count: int) -> List[str]:
    """Generates a specified number of responses using the Google Gemini API with rate limiting."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    responses: List[str] = []
    delay = INITIAL_DELAY # Устанавливаем начальную задержку
    errors = 0
    for _ in tqdm(range(generation_count), desc="Generating responses"):
        while True:
            try:
                response = model.generate_content(prompt)
                responses.append(response.text)
                time.sleep(delay) # Добавляем задержку между запросами
                break # Выходим из цикла, если запрос успешен
            except Exception as e:
                if "rateLimitExceeded" in str(e): # Проверяем, является ли ошибка превышением лимита
                    delay += 0.5 # Увеличиваем задержку
                    logger.warning(f"Rate limit exceeded. Increasing delay to {delay} seconds.")
                    time.sleep(delay) # Ждем перед повторным запросом
                    errors += 1
                else:
                    logger.error(f"Error generating response: {e}")
                    logger.error(traceback.format_exc())
                    break # Выходим из цикла при других ошибках
            if errors>20:
                break
    return responses

def extract_prompts_from_json(json_string: str) -> List[str]:
    """Extracts prompts from a JSON string."""
    try:
        if json_string.startswith("```json"):
            json_string = json_string[7:]  # Удаляем префикс "'''json"
        if json_string.endswith("```"):
            json_string = json_string[:-3]  # Удаляем суффикс "'''"

        data: List[Dict[str, str]] = json.loads(json_string)
        prompts: List[str] = [item["prompt"] for item in data]
        return prompts
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON string: {json_string}")
    except KeyError:
        raise KeyError(f"Missing 'prompt' key in JSON: {json_string}")
    except Exception as e:
        raise RuntimeError(f"Error extracting prompts from JSON: {e}")

def process_responses(responses: List[str]) -> List[str]:
    """Processes responses, extracting and flattening prompts."""
    all_prompts: List[str] = []
    for response in responses:
        try:
            extracted_prompts = extract_prompts_from_json(response)
            all_prompts.extend(extracted_prompts)
        except (ValueError, KeyError, RuntimeError) as e:
            logger.error(f"Error processing response: {e}, response: {response}")
    return all_prompts

def write_prompts_to_file(prompts: List[str], file_path: str) -> None:
    """Writes prompts to the specified file, one per line."""
    try:
        with open(file_path, "a", encoding="utf-8") as file:  # Используем режим "a" (append)
            for prompt in prompts:
                file.write(prompt + "\n")
    except Exception as e:
        raise RuntimeError(f"Error writing to file: {file_path}, {e}")

def write_raw_responses_to_file(responses: List[str], file_path: str) -> None:
    """Writes raw responses to the specified file, one per line."""
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            for response in responses:
                file.write(response + "\n")
    except Exception as e:
        raise RuntimeError(f"Error writing raw responses to file: {file_path}, {e}")

def main() -> None:
    """Main function to orchestrate the process."""
    try:
        api_key = load_api_key()
        prompt = read_prompt_from_file(PROMPT_FILE)
        responses = generate_responses(api_key, prompt, GENERATION_COUNT)
        write_raw_responses_to_file(responses, RAW_RESPONSES_FILE) # сохраняем ответы модели в файл
        extracted_prompts = process_responses(responses)
        write_prompts_to_file(extracted_prompts, OUTPUT_FILE)
        logger.info(f"Successfully processed {len(extracted_prompts)} prompts and wrote them to {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    logger.remove()
    logger.add("prompts.log", rotation="21 MB", retention=3, compression="zip", backtrace=True, diagnose=True)
    logger.add("prompts_ERROR.log", rotation="20 MB", retention=3, compression="zip", backtrace=True,
               diagnose=True, level='ERROR')
    try:
        logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> <level>{message}</level>",
                   level='INFO')
    except Exception as e:
        logger.debug(f'logger.add(sys.stdout) Error: {str(e)}')

    logger.info('Start Prompts gen')

    main()