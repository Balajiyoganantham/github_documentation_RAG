# Setup Guide

## Environment Configuration

This application uses a `.env` file to securely store your Groq API key.

### Step 1: Create .env file

Create a file named `.env` in the project root directory (same level as `app.py`) with the following content:

```bash
GROQ_API_KEY=your_actual_groq_api_key_here
```

### Step 2: Get your Groq API Key

1. Go to [Groq Console](https://console.groq.com/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the API key

### Step 3: Update .env file

Replace `your_actual_groq_api_key_here` with your actual Groq API key:

```bash
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Step 4: Run the application

```bash
streamlit run app.py
```

## Security Notes

- The `.env` file is already included in `.gitignore` to prevent accidentally committing your API key
- Never share your API key publicly
- The application will automatically load the API key from the `.env` file on startup

## Alternative Configuration

If you prefer not to use a `.env` file, you can also:
1. Set the environment variable directly in your terminal
2. Enter the API key manually in the Streamlit sidebar each time you run the app 