# API Keys Setup

## Required API Keys

This agent requires **two free API keys** to function:

### 1. Google Gemini API Key

Used for: LLM inference (planning, evaluation, summarization, report writing) + text embeddings.

**Get it free at:** https://aistudio.google.com

Steps:
1. Sign in with your Google account at https://aistudio.google.com
2. Click **"Get API key"** in the left sidebar
3. Create a new project (or select existing)
4. Copy your API key

**Free tier:** Generous free quota; no credit card required.

### 2. Tavily Search API Key

Used for: Real-time web search (retrieves URLs for each sub-question).

**Get it free at:** https://tavily.com

Steps:
1. Sign up at https://app.tavily.com/sign-up
2. Verify your email
3. Copy the API key from your dashboard

**Free tier:** 1,000 searches/month — more than enough for dozens of research reports.

---

## Adding Keys to the Project

1. Copy the example env file:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and replace the placeholder values:
   ```
   GEMINI_API_KEY=your-actual-gemini-key-here
   TAVILY_API_KEY=tvly-your-actual-tavily-key-here
   ```

3. **Never commit `.env` to git** — it is already in `.gitignore`.

---

## Verifying Your Keys

Run the config smoke test:
```bash
python -c "from app.config.settings import get_settings; s = get_settings(); print('Keys OK')"
```

If keys are invalid, you will see a descriptive validation error.
