import argparse
import asyncio
import logging
import os
import sys
import json
import secrets
from pathlib import Path
from aiohttp import web
import torch

from moshi.server import ServerState, seed_all
from moshi.models import loaders
from moshi.client_utils import log
from huggingface_hub import hf_hub_download

# Define handlers
async def health_check(request):
    state = request.app.get("state")
    return web.json_response({
        "status": "healthy", 
        "model_type": state.model_type if state else "unknown",
        "device": str(state.device) if state else "unknown"
    })

async def api_docs(request):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Personaplex API Docs</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; }
            code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
            .endpoint { border: 1px solid #ddd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .method { font-weight: bold; color: #007bff; }
        </style>
    </head>
    <body>
        <h1>Personaplex (Moshi) API</h1>
        
        <div class="endpoint">
            <span class="method">GET</span> <code>/health</code>
            <p>Returns server health status and loaded model info.</p>
        </div>

        <div class="endpoint">
            <span class="method">WS</span> <code>/api/chat</code>
            <p>WebSocket endpoint for real-time speech-to-speech.</p>
            <ul>
                <li><strong>Protocol:</strong> Binary</li>
                <li><strong>Input:</strong> PCM Audio (Opus encoded) - Prefix <code>\\x01</code></li>
                <li><strong>Output:</strong> 
                    <ul>
                        <li>Audio: Prefix <code>\\x01</code> + Opus bytes</li>
                        <li>Text: Prefix <code>\\x02</code> + UTF-8 bytes</li>
                    </ul>
                </li>
            </ul>
        </div>
    </body>
    </html>
    """
    return web.Response(text=html_content, content_type="text/html")

def main():
    parser = argparse.ArgumentParser(description="Custom Personaplex Server")
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--hf-repo", type=str, default="nvidia/personaplex-7b-v1")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Args expected by moshi logic (replicating defaults)
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--lora-weight", type=str, help="Path to a local checkpoint file for LoRA.", default=None)
    parser.add_argument("--config-path", type=str, help="Path to a local config file.", default=None)
    parser.add_argument("--cfg-coef", type=float, default=1., help="CFG coefficient.")
    parser.add_argument("--no_fuse_lora", action="store_false", dest="fuse_lora", default=True)
    parser.add_argument("--half", action="store_const", const=torch.float16, default=torch.bfloat16, dest="dtype")
    
    args = parser.parse_args()
    
    seed_all(42424242)
    
    log("info", f"Starting Custom Server on {args.host}:{args.port}")
    
    # Load Model
    log("info", f"Retrieving checkpoint from {args.hf_repo}")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo, args.moshi_weight, args.mimi_weight, args.tokenizer,
        lora_weights=args.lora_weight, config_path=args.config_path
    )
    
    log("info", "Loading Mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    
    log("info", "Loading Moshi")
    lm = checkpoint_info.get_moshi(device=args.device, dtype=args.dtype, fuse_lora=args.fuse_lora)
    
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    
    state = ServerState(checkpoint_info.model_type, mimi, text_tokenizer, lm, args.cfg_coef, args.device,
                        **checkpoint_info.lm_gen_config)
    
    log("info", "Warming up model")
    state.warmup()
    
    # Setup App
    app = web.Application()
    app["state"] = state
    
    # Routes
    app.router.add_get("/health", health_check)
    app.router.add_get("/api/docs", api_docs)
    app.router.add_get("/api/chat", state.handle_chat)
    
    # Root redirect to docs
    async def root_handler(request):
        raise web.HTTPFound('/api/docs')
    app.router.add_get("/", root_handler)
    
    log("info", "Server Ready")
    web.run_app(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
