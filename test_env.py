#!/usr/bin/env python3
"""
Test script to check if GROQ_API_KEY can be accessed from .env file
"""

import os
from dotenv import load_dotenv

def test_env_loading():
    """Test if environment variables are loaded correctly"""
    print("🔍 Testing .env file loading...")
    
    # Check if .env file exists
    env_file_path = ".env"
    if os.path.exists(env_file_path):
        print(f"✅ .env file found at: {os.path.abspath(env_file_path)}")
        
        # Read .env file content
        try:
            with open(env_file_path, 'r') as f:
                content = f.read().strip()
                print(f"📄 .env file content: {content}")
        except Exception as e:
            print(f"❌ Error reading .env file: {e}")
    else:
        print(f"❌ .env file not found at: {os.path.abspath(env_file_path)}")
    
    # Load environment variables
    print("\n🔄 Loading environment variables...")
    load_dotenv()
    
    # Check GROQ_API_KEY
    groq_api_key = os.getenv('GROQ_API_KEY')
    if groq_api_key:
        print(f"✅ GROQ_API_KEY found: {groq_api_key[:10]}...{groq_api_key[-10:]}")
        
        # Check if it's a valid format
        if groq_api_key.startswith('gsk_'):
            print("✅ GROQ_API_KEY format looks valid (starts with 'gsk_')")
        else:
            print("⚠️  GROQ_API_KEY format might be invalid (should start with 'gsk_')")
    else:
        print("❌ GROQ_API_KEY not found in environment variables")
    
    # Check other environment variables
    print("\n📋 All environment variables:")
    for key, value in os.environ.items():
        if 'GROQ' in key.upper() or 'API' in key.upper():
            if value and len(value) > 10:
                print(f"  {key}: {value[:10]}...{value[-10:]}")
            else:
                print(f"  {key}: {value}")
    
    # Test the exact same code as in rag_system.py
    print("\n🧪 Testing rag_system.py style loading...")
    try:
        GROQ_API_KEY = os.getenv('GROQ_API_KEY')
        if not GROQ_API_KEY:
            print("❌ GROQ_API_KEY not found - same error as in rag_system.py")
        else:
            print("✅ GROQ_API_KEY found - should work in rag_system.py")
            
    except Exception as e:
        print(f"❌ Error in rag_system.py style loading: {e}")

if __name__ == "__main__":
    test_env_loading() 