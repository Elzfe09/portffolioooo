from pydantic import BaseModel
from typing import Annotated, List, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pyjokes import get_joke
import random

# -----------------------------
# Model: represent a joke
# -----------------------------
class Joke(BaseModel):
    text: str
    category: str
    approved: bool = False  # new field for critic’s decision


# -----------------------------
# State: what the bot remembers
# -----------------------------
class JokeState(BaseModel):
    jokes: List[Joke] = []
    jokes_choice: Literal["n", "c", "q"] = "n"
    category: str = "neutral"
    language: str = "en"
    quit: bool = False

# -----------n------------------
# Menu node
# -----------------------------
def show_menu(state: JokeState) -> dict:
    user_input = input("\n[n] Next  [c] Category  [q] Quit\n> ").strip().lower()
    if user_input not in ["n", "c", "q"]:
        print("Invalid input. Defaulting to 'n'.")
        user_input = "n"
    return {"jokes_choice": user_input}


# -----------------------------
# Writer node: generates joke
# -----------------------------
def writer_node(state: JokeState) -> dict:
    category = state.category
    if category == "all":
        category = random.choice(["neutral", "chuck"])
    joke_text = get_joke(language=state.language, category=category)
    new_joke = Joke(text=joke_text, category=category)
    print(f"\n🖋️ Writer wrote a joke in category [{category}]...")
    return {"jokes": [new_joke]}


# -----------------------------
# Critic node: evaluate joke
# -----------------------------
def critic_node(state: JokeState) -> dict:
    if not state.jokes:
        return {}
    last_joke = state.jokes[-1]
    # simple evaluation — reject if joke too short
    if len(last_joke.text) < 40:
        print("🧠 Critic: This joke is too short, rejecting it.")
        last_joke.approved = False
        return {"jokes": [last_joke]}
    else:
        print("🧠 Critic: Approved this joke!")
        last_joke.approved = True
        print(f"\n🤣 {last_joke.text}")
        return {"jokes": [last_joke]}


# -----------------------------
# Category update
# -----------------------------
def update_category(state: JokeState) -> dict:
    categories = ["neutral", "chuck", "all"]
    while True:
        try:
            selection = int(input("Select category [0=neutral,1=chuck,2=all]: ").strip())
            if 0 <= selection < len(categories):
                break
            print("Invalid selection.")
        except ValueError:
            print("Please enter a number.")
    print(f"✅ Category changed to {categories[selection]}")
    return {"category": categories[selection]}


# -----------------------------
# Exit
# -----------------------------
def exit_bot(state: JokeState) -> dict:
    print("\n👋 Exiting bot. Thanks for laughing!")
    return {"quit": True}


# -----------------------------
# Router logic
# -----------------------------
def route_choice(state: JokeState) -> str:
    if state.jokes_choice == "n":
        return "writer"
    elif state.jokes_choice == "c":
        return "update_category"
    elif state.jokes_choice == "q":
        return "exit_bot"
    return "exit_bot"


# -----------------------------
# Build the graph
# -----------------------------
def build_joke_graph() -> CompiledStateGraph:
    workflow = StateGraph(JokeState)

    workflow.add_node("show_menu", show_menu)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("update_category", update_category)
    workflow.add_node("exit_bot", exit_bot)

    workflow.set_entry_point("show_menu")

    workflow.add_conditional_edges(
        "show_menu",
        route_choice,
        {
            "writer": "writer",
            "update_category": "update_category",
            "exit_bot": "exit_bot",
        },
    )

    # After writer -> critic -> back to menu
    workflow.add_edge("writer", "critic")
    workflow.add_edge("critic", "show_menu")
    workflow.add_edge("update_category", "show_menu")
    workflow.add_edge("exit_bot", END)

    return workflow.compile()


# -----------------------------
# Main
# -----------------------------
def main():
    print("🎭 Welcome to the Writer & Critic Joke Bot!")
    graph = build_joke_graph()
    final_state = graph.invoke(JokeState(), config={"recursion_limit": 100})

    print("\nHere are all jokes you got:\n")
    for i, joke in enumerate(final_state.jokes, 1):
        status = "✅ Approved" if joke.approved else "❌ Rejected"
        print(f"{i}. [{joke.category}] {status} - {joke.text}")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    main()
