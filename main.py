"""
Command-line AI Agent for Daily Diet Platform
=============================================
An interactive agent that uses OpenAI GPT-4 as the LLM backbone,
with MCP (Model Context Protocol) tools for querying meal data.

The agent follows a reasoning-and-action loop:
  1. Receive user input
  2. Use the LLM to determine actions
  3. Execute MCP tools or return a final response
  4. Repeat as needed
"""

from datetime import datetime
import os
import sys
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CREDENTIALS_FILE = "credentials.json"
INPUT_DIR = "./input"
INPUT_FILE = os.path.join(INPUT_DIR, "calories.json")
USER_ID_FILE = "USER_ID.md"
DATA_URL = "https://git.toptal.com/screeners/calories-json/-/raw/main/calories.json"

MENU_OPTIONS = {
    "1": "Provide a list of the meal history, sorted by date and time in descending order and show only the first 3 lines and the last 3 lines",
    "2": "Recommend 5 meals for the upcoming days to ensure the diet is healthy and non-repetitive, giving preference to the user's favorites",
    "3": "Provide a list of the protein consumption for a specified month and year for a given user",
    "c": "Change USER_ID",
    "q": "Quit",
}


# ─────────────────────────────────────────────
# MCP Tool Registry
# ─────────────────────────────────────────────
class MCPToolRegistry:
    """
    A lightweight MCP (Model Context Protocol) tool registry.
    Tools are registered as callable functions with metadata
    that can be exposed to the LLM as function definitions.
    """

    def __init__(self):
        self._tools: dict = {}

    def register(self, name: str, description: str, parameters: dict):
        """Decorator to register a function as an MCP tool."""
        def decorator(func):
            self._tools[name] = {
                "function": func,
                "definition": {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    },
                },
            }
            return func
        return decorator

    def get_definitions(self) -> list:
        """Return OpenAI-compatible tool definitions for all registered tools."""
        return [t["definition"] for t in self._tools.values()]

    def execute(self, name: str, arguments: dict):
        """Execute a registered tool by name with the given arguments."""
        if name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            result = self._tools[name]["function"](**arguments)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": str(exc)})


# Create global tool registry
tools = MCPToolRegistry()


# ─────────────────────────────────────────────
# MCP Tool: get_user_meals
# ─────────────────────────────────────────────
@tools.register(
    name="get_user_meals",
    description=(
        "Access the JSON file in the ./input directory and filter all meals "
        "for a specified user_id. Returns a compact summary of the user's "
        "meal data including: user profile, favorite meals, unique meals "
        "with average nutrition, meal type distribution, and the 10 most "
        "recent meals. Optimized to minimize token usage."
    ),
    parameters={
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "The user ID to filter meals for",
            }
        },
        "required": ["user_id"],
    },
)
def get_user_meals(user_id: str) -> dict:
    """MCP tool that reads the JSON data file and filters meals by user_id.
    Returns a compact summary instead of raw data to stay within token limits.
    """
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            all_meals = json.load(f)
    except FileNotFoundError:
        return {"error": f"Input file not found at {INPUT_FILE}"}
    except json.JSONDecodeError:
        return {"error": "Failed to parse the input JSON file"}

    # Filter meals for the given user_id (data stores user_id as string)
    user_meals = [
        meal for meal in all_meals if str(meal.get("user_id")) == str(user_id)
    ]

    if not user_meals:
        return {"error": f"No meals found for user_id {user_id}"}

    # ── User profile (from first record) ──
    sample = user_meals[0]
    profile = {
        "user_id": str(user_id),
        "age": sample.get("age"),
        "user_weight": sample.get("user_weight"),
    }

    # ── Favorite meals ──
    favorites = list({
        m["name"] for m in user_meals
        if str(m.get("favorite")).lower() == "true"
    })

    # ── Unique meals with average nutritional info ──
    meal_stats: dict = {}
    for m in user_meals:
        name = m["name"]
        if name not in meal_stats:
            meal_stats[name] = {
                "count": 0, "calories": 0, "fat": 0, "carbs": 0, "protein": 0,
                "types": set(), "procedence": set(),
            }
        s = meal_stats[name]
        s["count"] += 1
        s["calories"] += float(m.get("calories", 0))
        s["fat"] += float(m.get("fat", 0))
        s["carbs"] += float(m.get("carbs", 0))
        s["protein"] += float(m.get("protein", 0))
        s["types"].add(m.get("type", ""))
        s["procedence"].add(m.get("procedence", ""))

    unique_meals = []
    for name, s in meal_stats.items():
        c = s["count"]
        unique_meals.append({
            "name": name,
            "times_eaten": c,
            "avg_cal": round(s["calories"] / c),
            "avg_fat": round(s["fat"] / c, 1),
            "avg_carbs": round(s["carbs"] / c, 1),
            "avg_protein": round(s["protein"] / c, 1),
            "is_favorite": name in favorites,
            "types": list(s["types"]),
            "procedence": list(s["procedence"]),
        })

    # ── Meal type distribution ──
    type_counts: dict = {}
    for m in user_meals:
        t = m.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    # ── 10 most recent meals (compact) ──
    sorted_meals = sorted(
        user_meals,
        key=lambda m: (m["date_consumed"], m["time_consumed"]),
        reverse=True,
    )
    recent = [
        {
            "name": m["name"],
            "type": m["type"],
            "date": m["date_consumed"],
            "calories": m["calories"],
            "favorite": m["favorite"],
            "procedence": m["procedence"],
        }
        for m in sorted_meals[:10]
    ]

    return {
        "profile": profile,
        "total_meals": len(user_meals),
        "favorites": favorites,
        "unique_meals": unique_meals,
        "type_distribution": type_counts,
        "recent_meals": recent,
    }

@tools.register(
    name="get_protein_consumption",
    description=(
        "Get protein consumption for a specified user, month, and year. "
        "Returns per-meal protein breakdown sorted by date/time descending, "
        "plus the total protein consumed that month."
    ),
    parameters={
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "The user ID to filter meals for",
            },
            "month": {
                "type": "integer",
                "description": "Month number (1-12)",
            },
            "year": {
                "type": "integer",
                "description": "Year (e.g. 2022)",
            },
        },
        "required": ["user_id", "month", "year"],
    },
)
def get_protein_consumption(user_id: str, month: int, year: int) -> dict:
    """MCP tool that returns protein consumption per meal for a given user/month/year."""
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            all_meals = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"error": "Could not load meal data."}

    user_meals = [m for m in all_meals if str(m.get("user_id")) == str(user_id)]

    if not user_meals:
        return {"error": f"No meals found for USER_ID {user_id}."}

    # Filter by month and year
    meals_in_month = [
        m for m in user_meals
        if datetime.strptime(m["date_consumed"], "%Y-%m-%d").month == month
        and datetime.strptime(m["date_consumed"], "%Y-%m-%d").year == year
    ]

    if not meals_in_month:
        return {
            "user_id": str(user_id),
            "month": month,
            "year": year,
            "total_protein": 0,
            "meals": [],
            "message": f"No meals found for {year}-{month:02d}.",
        }

    # Sort by date/time descending
    meals_in_month.sort(
        key=lambda m: (m["date_consumed"], m["time_consumed"]),
        reverse=True,
    )

    total_protein = round(sum(float(m["protein"]) for m in meals_in_month), 2)

    # Per-meal breakdown (compact)
    breakdown = [
        {
            "date": m["date_consumed"],
            "time": m["time_consumed"],
            "name": m["name"],
            "type": m["type"],
            "protein": float(m["protein"]),
        }
        for m in meals_in_month
    ]

    return {
        "user_id": str(user_id),
        "month": month,
        "year": year,
        "total_protein": total_protein,
        "meal_count": len(breakdown),
        "meals": breakdown,
    }

# ─────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────
def load_credentials() -> dict:
    """Load and validate the credentials.json file."""
    if not os.path.exists(CREDENTIALS_FILE):
        print(f"❌ Error: {CREDENTIALS_FILE} not found.")
        sys.exit(1)

    try:
        with open(CREDENTIALS_FILE, "r") as f:
            creds = json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        print(f"❌ Error reading {CREDENTIALS_FILE}: {exc}")
        sys.exit(1)

    required_keys = ["provider", "model", "api_key_env"]
    for key in required_keys:
        if key not in creds:
            print(f"❌ Error: Missing '{key}' in {CREDENTIALS_FILE}")
            sys.exit(1)

    return creds


def establish_llm_connection(creds: dict) -> tuple[OpenAI, str, float]:
    """Establish connection with the LLM using credentials."""
    load_dotenv()

    api_key_env = creds["api_key_env"]
    api_key = os.getenv(api_key_env)

    if not api_key:
        print(f"❌ Error: Environment variable '{api_key_env}' not set.")
        print("   Make sure it is defined in your .env file.")
        sys.exit(1)

    model = creds.get("model", "gpt-4")
    temperature = creds.get("temperature", 0.7)

    try:
        client = OpenAI(api_key=api_key)
        # Quick validation call
        client.models.list()
        print(f"✅ Connected to OpenAI — model: {model}")
    except Exception as exc:
        print(f"❌ Failed to connect to LLM: {exc}")
        sys.exit(1)

    return client, model, temperature


def ensure_input_data() -> None:
    """Verify that the input JSON file exists; download if not."""
    if os.path.exists(INPUT_FILE):
        print(f"✅ Input data found at {INPUT_FILE}")
        return

    print(f"⬇️  Input file not found. Downloading from {DATA_URL} ...")
    os.makedirs(INPUT_DIR, exist_ok=True)

    try:
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        with open(INPUT_FILE, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"✅ Downloaded and saved to {INPUT_FILE}")
    except requests.RequestException as exc:
        print(f"❌ Failed to download input data: {exc}")
        sys.exit(1)

    # Validate it's parseable JSON
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            json.load(f)
    except json.JSONDecodeError:
        print("❌ Downloaded file is not valid JSON.")
        sys.exit(1)


def prompt_user_id() -> str:
    """Prompt the user for a USER_ID and persist it in USER_ID.md."""
    user_id = input("\n🔑 Enter your USER_ID: ").strip()
    if not user_id:
        print("❌ USER_ID cannot be empty.")
        sys.exit(1)

    with open(USER_ID_FILE, "w") as f:
        f.write(f"# User ID\n\n{user_id}\n")
    print(f"✅ USER_ID '{user_id}' saved to {USER_ID_FILE}")
    return user_id


# ─────────────────────────────────────────────
# Agent Logic
# ─────────────────────────────────────────────
def format_meal_history(user_id: str) -> str:
    """
    Build the formatted meal history for a user directly,
    sorted by date and time descending, showing first 3 and last 3 lines.
    This optimizes token usage by filtering in code instead of the LLM.

    Output format: <date_consumed> <time_consumed> <type> <procedence> <name>
    """
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            all_meals = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return "Error: Could not load meal data."

    user_meals = [
        m for m in all_meals if str(m.get("user_id")) == str(user_id)
    ]

    if not user_meals:
        return f"No meals found for USER_ID {user_id}."

    # Sort by date descending, then time descending
    user_meals.sort(
        key=lambda m: (m["date_consumed"], m["time_consumed"]),
        reverse=True,
    )

    def fmt(meal: dict) -> str:
        return (
            f"{meal['date_consumed']} {meal['time_consumed']} "
            f"{meal['type']} {meal['procedence']} {meal['name']}"
        )

    lines = [fmt(m) for m in user_meals]

    if len(lines) <= 6:
        return "\n".join(lines)

    # Show first 3 and last 3 with ellipsis
    display = lines[:3] + ["..."] + lines[-3:]
    return "\n".join(display)


def _build_system_prompt(user_id: str) -> str:
    """Build the system prompt for the given user."""
    return (
        "You are a helpful dietary assistant AI agent. "
        f"You are assisting USER_ID {user_id}. "
        "You have access to a tool called 'get_user_meals' that retrieves "
        "all meal records for a given user from the diet platform database. "
        "You also have access to a function get_protein_consumption_by_month_and_year(user_id: str, month: int, year: int) that retrieves "
        "the protein consumption for a specified month and year for a given user. "
        "When analyzing diets, consider nutritional balance (calories, fat, "
        "carbs, protein), meal timing, variety, and the user's favorites. "
        "Be concise, helpful, and format your responses clearly."
    )


def run_agent_loop(
    client: OpenAI,
    model: str,
    temperature: float,
    user_id: str,
) -> None:
    """
    Main agent loop.
    Presents a menu, takes user selection, and either:
      - Handles option 1 (history) with direct code filtering + LLM formatting
      - Handles option 2 (recommendations) via full LLM reasoning with MCP tools
      - Handles option 3 (protein consumption) with direct code filtering + LLM formatting
      - Handles option c (change user) to switch USER_ID without restarting
      - Quits on 'q'
    """
    conversation_history = [{"role": "system", "content": _build_system_prompt(user_id)}]

    print("\n" + "=" * 60)
    print(f"🍽️  Diet Agent — Ready (USER_ID: {user_id})")
    print("=" * 60)

    while True:
        # Display menu
        print("\nSelect an option:")
        for key, text in MENU_OPTIONS.items():
            print(f"  {key}. {text}")

        choice = input("\n👉 Your choice: ").strip().lower()

        if choice == "q":
            print("\n👋 Goodbye! Stay healthy!")
            break

        if choice == "c":
            new_id = input("\n🔑 Enter new USER_ID: ").strip()
            if not new_id:
                print("⚠️  USER_ID cannot be empty.")
                continue
            user_id = new_id
            with open(USER_ID_FILE, "w") as f:
                f.write(f"# User ID\n\n{user_id}\n")
            # Reset conversation for the new user
            conversation_history = [{"role": "system", "content": _build_system_prompt(user_id)}]
            print(f"✅ Switched to USER_ID '{user_id}' — conversation reset.")
            continue

        if choice not in MENU_OPTIONS:
            print("⚠️  Invalid option. Please try again.")
            continue

        if choice == "1":
            # Option 1: Meal history — filtered in code to save tokens
            print("\n📋 Meal History (sorted by date/time descending):\n")
            history_output = format_meal_history(user_id)
            print(history_output)

            # Also send a brief summary to the LLM for natural language wrap-up
            user_message = (
                f"Here is the meal history for user {user_id}, "
                f"sorted by date and time descending (first 3 and last 3):\n\n"
                f"{history_output}\n\n"
                "Briefly acknowledge this list. Do not repeat it."
            )
            conversation_history.append({"role": "user", "content": user_message})

            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=conversation_history,
                )
                assistant_msg = response.choices[0].message.content
                conversation_history.append({"role": "assistant", "content": assistant_msg})
                print(f"\n🤖 Agent: {assistant_msg}")
            except Exception as exc:
                print(f"\n⚠️  LLM error: {exc}")

        elif choice == "3":
            # Option 3: Protein consumption — filtered in code + LLM summary
            print(f"\n📋 Protein consumption for user {user_id}:\n")
            month_str = input("Enter month (1-12): ").strip()
            year_str = input("Enter year (e.g. 2022): ").strip()

            # Validate inputs
            try:
                month = int(month_str)
                year = int(year_str)
                if not (1 <= month <= 12):
                    raise ValueError
            except ValueError:
                print("⚠️  Invalid month or year. Please enter valid numbers.")
                continue

            result = get_protein_consumption(
                user_id=user_id, month=month, year=year
            )

            if isinstance(result, dict) and "error" in result:
                print(f"⚠️  {result['error']}")
                continue

            # Display per-meal breakdown
            meals = result.get("meals", [])
            if not meals:
                print(f"No meals found for {year}-{month:02d}.")
                continue

            for m in meals:
                print(f"  {m['date']} {m['time']}  {m['type']:<10} {m['name']:<30} {m['protein']:.1f}g")
            print(f"\n  Total protein: {result['total_protein']:.1f}g across {result['meal_count']} meals")

            # Send compact summary to LLM
            user_message = (
                f"User {user_id} consumed {result['total_protein']:.1f}g of protein "
                f"across {result['meal_count']} meals in {year}-{month:02d}. "
                "Briefly comment on whether this protein intake is adequate. Do not list meals."
            )
            conversation_history.append({"role": "user", "content": user_message})

            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=conversation_history,
                )
                assistant_msg = response.choices[0].message.content
                conversation_history.append({"role": "assistant", "content": assistant_msg})
                print(f"\n🤖 Agent: {assistant_msg}")
            except Exception as exc:
                print(f"\n⚠️  LLM error: {exc}")

        elif choice == "2":
            # Option 2: Meal recommendations — full LLM reasoning with MCP tools
            user_message = (
                f"Using the get_user_meals tool, retrieve the meal data for user_id '{user_id}'. "
                "Then analyze the user's dietary history and recommend 5 meals for the upcoming days. "
                "The recommendations should:\n"
                "- Ensure a healthy and balanced diet\n"
                "- Avoid being repetitive (suggest variety)\n"
                "- Give preference to the user's favorite meals\n"
                "- Consider the user's nutritional profile (calories, fat, carbs, protein)\n"
                "Format each recommendation with meal name, suggested type (breakfast/lunch/dinner/snack), "
                "and a brief reason why it's recommended."
            )

            conversation_history.append({"role": "user", "content": user_message})

            # Reasoning-and-action loop: keep calling the LLM until we get a final response
            max_iterations = 10
            for _ in range(max_iterations):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        temperature=temperature,
                        messages=conversation_history,
                        tools=tools.get_definitions(),
                        tool_choice="auto",
                    )
                except Exception as exc:
                    print(f"\n⚠️  LLM error: {exc}")
                    break

                message = response.choices[0].message

                # If there are tool calls, execute them and feed results back
                if message.tool_calls:
                    conversation_history.append(message)
                    for tool_call in message.tool_calls:
                        fn_name = tool_call.function.name
                        try:
                            fn_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            fn_args = {}

                        print(f"  🔧 Calling tool: {fn_name}({fn_args})")
                        result = tools.execute(fn_name, fn_args)

                        conversation_history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result,
                            }
                        )
                    # Continue the loop — let the LLM process tool results
                    continue

                # No tool calls — this is the final response
                if message.content:
                    conversation_history.append(
                        {"role": "assistant", "content": message.content}
                    )
                    print(f"\n🤖 Agent:\n{message.content}")
                break

        # Keep conversation history manageable (last 20 messages + system)
        if len(conversation_history) > 21:
            conversation_history = [conversation_history[0]] + conversation_history[-20:]


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────
def main():
    """
    Agent startup sequence:
    1. Check credentials.json and connect to LLM
    2. Verify/download input JSON data
    3. Prompt for USER_ID
    4. Enter the main agent loop
    """
    print("=" * 60)
    print("🚀 Diet Agent — Initializing...")
    print("=" * 60)

    # Step 1: Load credentials and connect to LLM
    print("\n[1/3] Checking credentials and connecting to LLM...")
    creds = load_credentials()
    client, model, temperature = establish_llm_connection(creds)

    # Step 2: Ensure input data is available
    print("\n[2/3] Checking input data...")
    ensure_input_data()

    # Step 3: Prompt for USER_ID
    print("\n[3/3] User identification...")
    user_id = prompt_user_id()

    # Enter the main agent loop
    run_agent_loop(client, model, temperature, user_id)


if __name__ == "__main__":
    main()
