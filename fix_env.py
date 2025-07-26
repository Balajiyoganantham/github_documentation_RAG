#!/usr/bin/env python3
"""
Script to fix the .env file by removing BOM character
"""

def fix_env_file():
    """Remove BOM from .env file"""
    env_file_path = ".env"
    
    try:
        # Read the file with UTF-8 encoding
        with open(env_file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        
        print(f"📄 Original content: {repr(content)}")
        
        # Remove BOM if present
        if content.startswith('\ufeff'):
            content = content[1:]
            print("🔧 BOM character removed")
        
        # Write back without BOM
        with open(env_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Fixed content: {repr(content)}")
        print("✅ .env file has been fixed!")
        
        # Test if it works now
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        groq_api_key = os.getenv('GROQ_API_KEY')
        if groq_api_key:
            print(f"✅ GROQ_API_KEY now accessible: {groq_api_key[:10]}...{groq_api_key[-10:]}")
        else:
            print("❌ GROQ_API_KEY still not accessible")
            
    except Exception as e:
        print(f"❌ Error fixing .env file: {e}")

if __name__ == "__main__":
    fix_env_file() 