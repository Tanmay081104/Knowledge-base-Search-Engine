"""Simple test of Groq integration"""

import os
from groq import Groq

# Test the Groq API key
def test_groq():
    try:
        client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Test a simple completion
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": "Say 'Hello from Groq!' if this is working correctly."}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        print("🎉 Groq Integration Test SUCCESSFUL!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Groq Integration Test FAILED: {str(e)}")
        return False

if __name__ == "__main__":
    test_groq()