#!/usr/bin/env python3
"""
DeepSeek R1 Web API - Complete Single File Application
Auto-installs dependencies, loads model to GPU, provides streaming API
"""

import os
import sys
import subprocess
import json
import asyncio
from typing import Dict, List, Optional, AsyncGenerator
from datetime import datetime
import logging

# Auto-install dependencies
def install_dependencies():
    """Install required packages automatically"""
    packages = [
        "torch==2.1.0",
        "transformers==4.36.0", 
        "accelerate==0.24.0",
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "pydantic==2.5.0",
        "sse-starlette==1.6.5"
    ]
    
    print("🔧 Installing dependencies...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            sys.exit(1)
    print("🎉 All dependencies installed!")

# Check if packages are installed, if not install them
try:
    import torch
    import transformers
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import uvicorn
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    print("🚀 First run detected - installing dependencies...")
    install_dependencies()
    # Re-import after installation
    import torch
    import transformers
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import uvicorn
    from sse_starlette.sse import EventSourceResponse

from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_DIR = "./deepseek_models"
PERSONA_FILE = "persona.txt"
MAX_LENGTH = 2048
TEMPERATURE = 0.7

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    message: str
    role: Optional[str] = "user"
    thinking: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = True

class ChatResponse(BaseModel):
    thinking: Optional[str] = None
    content: str
    sources: List[str] = []
    knowledge: List[str] = []
    rules: List[str] = []
    timestamp: str
    model_info: Dict[str, str]

class DeepSeekAPI:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.persona = ""
        self.device = None
        
    def load_persona(self):
        """Load persona from txt file"""
        try:
            if os.path.exists(PERSONA_FILE):
                with open(PERSONA_FILE, 'r', encoding='utf-8') as f:
                    self.persona = f.read().strip()
                logger.info(f"✅ Loaded persona from {PERSONA_FILE}")
            else:
                # Create default persona file
                default_persona = """You are DeepSeek Assistant, an AI created by DeepSeek.

PERSONALITY:
- Helpful, harmless, and honest
- Analytical and thoughtful
- Curious about learning
- Patient and understanding

KNOWLEDGE:
- Broad knowledge across many domains
- Strong in mathematics, science, and programming
- Current knowledge cutoff: October 2023
- Multilingual capabilities

RULES:
- Always think step by step when requested
- Provide structured responses with clear reasoning
- Cite sources when making specific claims
- Be transparent about limitations
- Prioritize accuracy over speed

BEHAVIOR:
- Start with thinking/reasoning if requested
- Structure responses clearly
- Use examples when helpful
- Ask clarifying questions when needed"""
                
                with open(PERSONA_FILE, 'w', encoding='utf-8') as f:
                    f.write(default_persona)
                self.persona = default_persona
                logger.info(f"✅ Created default persona file: {PERSONA_FILE}")
        except Exception as e:
            logger.error(f"❌ Error loading persona: {e}")
            self.persona = "You are a helpful AI assistant."

    def setup_gpu(self):
        """Setup GPU configuration"""
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🎮 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            self.device = "cpu"
            logger.info("💻 Using CPU")
        
        return self.device

    def load_model(self):
        """Load DeepSeek model to GPU"""
        try:
            logger.info(f"📥 Loading model: {MODEL_NAME}")
            logger.info(f"📁 Model directory: {MODEL_DIR}")
            
            # Create model directory
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                cache_dir=MODEL_DIR,
                trust_remote_code=True
            )
            
            # Load model with GPU optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                cache_dir=MODEL_DIR,
                device_map="auto",
                torch_dtype=torch.float16,  # Half precision for 4GB GPU
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Check GPU memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"🎮 GPU Memory Used: {memory_used:.2f}GB")
            
            logger.info("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False

    def parse_persona(self):
        """Parse persona into structured components"""
        sections = {
            "knowledge": [],
            "rules": [],
            "sources": []
        }
        
        current_section = None
        for line in self.persona.split('\n'):
            line = line.strip()
            if line.upper().startswith('KNOWLEDGE:'):
                current_section = "knowledge"
            elif line.upper().startswith('RULES:'):
                current_section = "rules"
            elif line.upper().startswith('SOURCES:'):
                current_section = "sources"
            elif line.startswith('- ') and current_section:
                sections[current_section].append(line[2:])
        
        return sections

    async def generate_thinking(self, message: str) -> str:
        """Generate thinking/reasoning step"""
        thinking_prompt = f"""<thinking>
Let me think about this step by step:

User asked: {message}

I need to:
1. Understand what they're asking
2. Consider relevant knowledge
3. Structure a helpful response
4. Check for any limitations or clarifications needed
</thinking>"""
        
        return thinking_prompt

    async def generate_response_stream(self, message: str, thinking: bool = False, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        try:
            # Prepare system prompt with persona
            system_prompt = f"System: {self.persona}\n\n"
            
            # Add thinking if requested
            full_prompt = system_prompt
            if thinking:
                thinking_text = await self.generate_thinking(message)
                full_prompt += thinking_text + "\n\n"
            
            full_prompt += f"User: {message}\nAssistant:"
            
            # Tokenize
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
            
            # Generation parameters
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.1
            }
            
            # Generate streaming response
            generated_text = ""
            with torch.no_grad():
                for _ in range(generation_kwargs["max_new_tokens"]):
                    outputs = self.model(inputs)
                    logits = outputs.logits[0, -1, :]
                    
                    # Apply temperature
                    if generation_kwargs["temperature"] > 0:
                        logits = logits / generation_kwargs["temperature"]
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    # Decode token
                    token_text = self.tokenizer.decode(next_token, skip_special_tokens=True)
                    
                    # Check for end of generation
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    generated_text += token_text
                    yield token_text
                    
                    # Update inputs for next iteration
                    inputs = torch.cat([inputs, next_token.unsqueeze(0)], dim=-1)
                    
                    # Prevent memory overflow
                    if inputs.shape[1] > MAX_LENGTH:
                        inputs = inputs[:, -MAX_LENGTH//2:]
            
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            yield f"Error: {str(e)}"

    async def generate_response(self, message: str, thinking: bool = False, **kwargs) -> ChatResponse:
        """Generate complete response"""
        try:
            # Collect streaming response
            content = ""
            async for token in self.generate_response_stream(message, thinking, **kwargs):
                content += token
            
            # Parse persona components
            persona_components = self.parse_persona()
            
            # Create response
            response = ChatResponse(
                thinking=await self.generate_thinking(message) if thinking else None,
                content=content.strip(),
                sources=persona_components["sources"],
                knowledge=persona_components["knowledge"],
                rules=persona_components["rules"],
                timestamp=datetime.now().isoformat(),
                model_info={
                    "model": MODEL_NAME,
                    "device": self.device,
                    "gpu_memory": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB" if torch.cuda.is_available() else "N/A"
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Response generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize API
deepseek = DeepSeekAPI()
app = FastAPI(title="DeepSeek R1 API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("🚀 Starting DeepSeek R1 API...")
    
    # Load persona
    deepseek.load_persona()
    
    # Setup GPU
    deepseek.setup_gpu()
    
    # Load model
    if not deepseek.load_model():
        logger.error("❌ Failed to load model")
        sys.exit(1)
    
    logger.info("✅ API ready!")

@app.get("/")
async def root():
    """API info endpoint"""
    return {
        "name": "DeepSeek R1 API",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "device": deepseek.device,
        "endpoints": {
            "chat": "/chat",
            "stream": "/stream",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB"
        }
    else:
        gpu_info = {"gpu_available": False}
    
    return {
        "status": "healthy",
        "model_loaded": deepseek.model is not None,
        "device": deepseek.device,
        **gpu_info
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint - returns complete response"""
    if not deepseek.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    response = await deepseek.generate_response(
        message=request.message,
        thinking=request.thinking,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    
    return response

@app.post("/stream")
async def stream_chat(request: ChatRequest):
    """Streaming chat endpoint - real-time text generation"""
    if not deepseek.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    async def generate():
        try:
            # Send initial metadata
            metadata = {
                "type": "metadata",
                "thinking": await deepseek.generate_thinking(request.message) if request.thinking else None,
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            
            # Stream content
            async for token in deepseek.generate_response_stream(
                message=request.message,
                thinking=request.thinking,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                chunk = {
                    "type": "content",
                    "token": token
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Send completion
            completion = {
                "type": "done",
                "sources": deepseek.parse_persona()["sources"],
                "knowledge": deepseek.parse_persona()["knowledge"],
                "rules": deepseek.parse_persona()["rules"]
            }
            yield f"data: {json.dumps(completion)}\n\n"
            
        except Exception as e:
            error = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error)}\n\n"
    
    return EventSourceResponse(generate())

if __name__ == "__main__":
    print("🤖 DeepSeek R1 Web API")
    print("=" * 50)
    print(f"📁 Model Directory: {MODEL_DIR}")
    print(f"📋 Persona File: {PERSONA_FILE}")
    print(f"🎯 Model: {MODEL_NAME}")
    print("=" * 50)
    
    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
