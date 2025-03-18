from transformers import pipeline
import gradio as gr
import time
from dotenv import load_dotenv
import requests
import json
import re
import logging
import os
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, Any

load_dotenv()

# --------------------
# Configuration Classes
# --------------------
class AppConfig:
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.model = os.getenv("MODEL", "llama")
        self.model_path = "python/snowflake_classifier"
        self.max_retries = 3
        self.model_weight = 0.3
        self.zero_shot_model = "facebook/bart-large-mnli"
        
   

class LoggerSetup:
    @staticmethod
    def configure_loggers():
        # Error logger
        error_logger = logging.getLogger('error_logger')
        error_handler = logging.FileHandler('server_errors.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        error_logger.addHandler(error_handler)

        # Info logger
        info_logger = logging.getLogger('info_logger')
        info_handler = logging.FileHandler('server_info.log')
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        info_logger.addHandler(info_handler)

        logging.basicConfig(level=logging.DEBUG)

# --------------------
# Core Functionality Classes
# --------------------
class TextAnalyzer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.main_classifier = pipeline(
            "text-classification",
            model=config.model_path,
            tokenizer=config.model_path,
            truncation=True,
            max_length=512
        )
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model=config.zero_shot_model
        )

    def analyze_text(self, text: str) -> Dict[str, Any]:
        main_result = self.main_classifier(text)[0]
        zero_shot_result = self.zero_shot_classifier(
            text,
            candidate_labels=["offensive", "non-offensive"],
        )
        
        zs_scores = {
            zero_shot_result['labels'][0]: zero_shot_result['scores'][0],
            zero_shot_result['labels'][1]: zero_shot_result['scores'][1]
        }
        
        combined_offensive = (main_result['score'] if main_result['label'] == "offensive" 
                            else 1 - main_result['score']) * self.config.model_weight
        combined_offensive += zs_scores.get("offensive", 0) * (1 - self.config.model_weight)
        
        return {
            "text": text,
            "main_label": main_result['label'],
            "main_confidence": main_result['score'],
            "zero_shot_scores": zs_scores,
            "combined_offensive": combined_offensive,
            "combined_safe": 1 - combined_offensive
        }

class ResponseGenerator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger('info_logger')
        self.logger.info(f"Current Model Running: {config.model}")

    
    def _clean_message(self, message: str) -> str:
        cleaned = re.sub(r'\\.', '', message)
        return re.sub(r"[\"']", '', cleaned)
    
    def _make_api_request(self, url: str, headers: dict, payload: dict) -> str:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            error_logger = logging.getLogger('error_logger')
            error_msg = f"API request failed: {str(e)}"
            error_logger.error(error_msg)
            raise RuntimeError(error_msg)

class GPTResponseGenerator(ResponseGenerator):
    def generate_response(self, analysis: Dict[str, Any]) -> str:
        is_offensive = analysis['combined_safe'] < analysis['combined_offensive']
        
        system_prompt = (
            "I’m going to give you the content of an email I’m about to send to my friend. "
            "Please pretend to be my friend and write the most likely response they would send back, "
            "in the same tone they normally use when replying to me. "
            "Assume there is no conflict and that we have a good relationship."
            if is_offensive else
            "I’m going to give you the content of an email I’m about to send to my friend. "
            "The message is intentionally sarcastic, harsh, or possibly offensive. "
            "I want you to respond as if you were my friend reacting honestly and naturally to the message—"
            "whether they’d be angry, sarcastic back, defensive, hurt, or try to de-escalate. "
            "Be realistic and write their most likely response based on how a normal person would react "
            "in a close but strained friendship."
        )
        
        # self.logger.info(system_prompt)

        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis["text"]}
            ]
        }

        response = self._make_api_request(
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}"
            },
            payload=payload
        )
        return self._clean_message(response)

class LlamaResponseGenerator(ResponseGenerator):
    def generate_response(self, analysis: Dict[str, Any]) -> str:
        is_offensive = analysis['combined_safe'] < analysis['combined_offensive']
        
        message = (
            f"I’m going to give you the content of an email I’m about to send to my friend. "
            f"Content: '{analysis['text']}' Please pretend to be my friend and write the most likely response "
            f"they would send back, in the same tone they normally use when replying to me. "
            f"Assume there is no conflict and that we have a good relationship. Keep it short and concise."
            if is_offensive else
            f"I’m going to give you the content of an email I’m about to send to my friend. "
            f"Content: '{analysis['text']}' The message is intentionally sarcastic, harsh, or possibly offensive. "
            f"I want you to respond as if you were my friend reacting honestly and naturally to the message—"
            f"whether they’d be angry, sarcastic back, defensive, hurt, or try to de-escalate. "
            f"Be realistic and write their most likely response based on how a normal person would react "
            f"in a close but strained friendship. Keep it short and concise."
        )
        
        # self.logger.info(message)

        payload = {
            "messages": [{"role": "user", "content": message}]
        }

        response = self._make_api_request(
            url="https://ai.hackclub.com/chat/completions",
            headers={"Content-Type": "application/json"},
            payload=payload
        )
        return self._clean_message(response)


class APIService:
    def __init__(self, analyzer: TextAnalyzer, response_generator: ResponseGenerator):
        self.app = FastAPI()
        self.analyzer = analyzer
        self.response_generator = response_generator
        self.logger = logging.getLogger('error_logger')
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        class AnalysisResponse(BaseModel):
            response: str
            safe_for_snowflake: float
            offensive: float

        @self.app.get("/analyze")
        async def analyze_endpoint(text: str):
            retries = 0
            analysis = self.analyzer.analyze_text(text)
            
            while retries < self.response_generator.config.max_retries:
                try:
                    response_text = self.response_generator.generate_response(analysis)
                    return AnalysisResponse(
                        response=response_text,
                        safe_for_snowflake=analysis['combined_safe'],
                        offensive=analysis['combined_offensive']
                    )
                except Exception as e:
                    self.logger.error(f"Attempt {retries+1} failed: {str(e)}")
                    retries += 1
                    time.sleep(1)
            
            self.logger.error("Max retries reached, using fallback")
            return AnalysisResponse(
                response="Service temporarily unavailable",
                safe_for_snowflake=analysis['combined_safe'],
                offensive=analysis['combined_offensive']
            )

# --------------------
# Main Application
# --------------------
class Application:
    def __init__(self):
        LoggerSetup.configure_loggers()
        self.config = AppConfig()
        self.analyzer = TextAnalyzer(self.config)
        self.response_generator = self._create_response_generator()
        
    
    def _create_response_generator(self) -> ResponseGenerator:
        if self.config.model == "gbt":
            return GPTResponseGenerator(self.config)
        elif self.config.model == "llama":
            return LlamaResponseGenerator(self.config)
        raise ValueError(f"Unknown model type: {self.config.model}")
    
    def run_api(self):
        api_service = APIService(self.analyzer, self.response_generator)
        uvicorn.run(api_service.app, host="0.0.0.0", port=8000)
    
    def run_gradio(self):
        interface = GradioInterface(self.analyzer).create_interface()
        interface.launch()

if __name__ == "__main__":
    app = Application()
    app.run_api()