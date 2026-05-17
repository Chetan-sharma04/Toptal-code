# Diet AI Agent 🍽️🤖

A command-line AI assistant powered by OpenAI's GPT-4 and built using the Model Context Protocol (MCP) pattern. This agent interacts with daily diet platform data to provide personalized insights, meal history, meal recommendations, and protein consumption tracking.

## Features ✨

The agent runs in an interactive infinite loop and offers the following features:
1. **Meal History**: View your past meals, sorted chronologically, intelligently summarized to save LLM tokens.
2. **AI Meal Recommendations**: Receive 5 personalized meal recommendations for the upcoming days, optimized for health, variety, and your favorite foods.
3. **Protein Consumption Tracking**: Calculate and analyze your total and per-meal protein intake for any specific month and year.
4. **Dynamic User Switching**: Switch between different users (`USER_ID`) seamlessly without restarting the application.
5. **Auto-Data Management**: Automatically downloads the required JSON diet data if it's missing locally.
6. **Optimized Token Usage**: Implements smart filtering and data summarization before sending context to the LLM to prevent rate limits and save costs.

## Prerequisites 🛠️

- Python 3.14+
- An OpenAI API Key
- `uv` (recommended) or `pip` for dependency management

## Setup & Installation 🚀

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd "Toptal code"
   ```

2. **Environment Variables**:
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Configuration Files**:
   - `credentials.json`: Maps the agent to use your environment variables and specifies the model. (Auto-created or make sure it matches the structure below).
     ```json
     {
       "provider": "openai",
       "model": "gpt-4",
       "api_key_env": "OPENAI_API_KEY",
       "temperature": 0.7
     }
     ```
   - `config.yaml`: Contains general configuration (optional depending on use case).

4. **Install Dependencies**:
   Using `uv` (recommended):
   ```bash
   uv sync
   ```
   Or using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage 🏃‍♂️

Start the interactive agent:

```bash
uv run main.py
```
*(Or `python main.py` if your virtual environment is activated)*

### Interaction Flow:
1. On startup, the agent verifies credentials and connects to GPT-4.
2. It checks for the local dataset (`./input/calories.json`). If missing, it downloads it automatically.
3. It prompts for your `USER_ID`.
4. You are presented with an interactive menu to choose actions.
5. The agent executes tools locally and uses the LLM to reason and format responses naturally.

## Project Structure 📂

```text
.
├── .env                  # Environment variables (API Key)
├── credentials.json      # LLM credentials mapping
├── main.py               # Main agent logic and CLI loop
├── input/
│   └── calories.json     # Diet dataset (auto-downloaded)
├── USER_ID.md            # Automatically tracks active user context
├── pyproject.toml        # Project dependencies and metadata
└── README.md             # Project documentation
```

## How It Works (Under the Hood) 🧠

- **MCP Tools Registry**: Custom tools like `get_user_meals` and `get_protein_consumption` are registered as callable functions. The agent exposes their schemas to GPT-4 via Function Calling.
- **Reasoning-and-Action Loop**: When you request recommendations or analysis, the LLM decides which tools to call, processes the internal database results, and generates a cohesive, natural language response.