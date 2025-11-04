#!/usr/bin/env python3
"""
Script to list available Gemini models using the API key from the .env file.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai


def list_gemini_models():
    """
    List all available Gemini models using the API key from .env file.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the GEMINI_API_KEY from environment variables
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file")
        return
    
    # Configure the API with the key
    genai.configure(api_key=api_key)
    
    print("Available Gemini Models:")
    print("-" * 50)
    
    # List all available models
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"Model Name: {model.name}")
            print(f"  Version: {model.version}")
            print(f"  Display Name: {model.display_name}")
            print(f"  Description: {model.description}")
            print("-" * 50)


if __name__ == "__main__":
    list_gemini_models()