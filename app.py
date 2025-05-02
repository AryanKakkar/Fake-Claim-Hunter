import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from neo4j import GraphDatabase
import torch
import gradio as gr

# Load environment variables from .env file
load_dotenv()

# === Neo4j Connection ===

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")  # Fetch password from environment variable

driver = GraphDatabase.driver(uri, auth=(user, password))

# === Load LLM and Tokenizer ===
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# === Neo4j Query Function ===
def query_knowledge_graph(subject):
    with driver.session() as session:
        result = session.run("""
            MATCH (s:Entity {name: $subject})-[r:RELATION]->(o:Entity)
            RETURN s.name AS subject, r.label AS predicate, o.name AS object
        """, subject=subject)
        return [(record["subject"], record["predicate"], record["object"]) for record in result]

# === Claim Checking Function ===
def check_claim_with_kg(claim):
    subj = claim.split(" is ")[0].strip()
    triplets = query_knowledge_graph(subj)

    if not triplets:
        context = f"No knowledge found about {subj} in the knowledge graph."
        facts_section = context
    else:
        facts_section = "Knowledge Graph Facts:\n"
        for s, p, o in triplets:
            facts_section += f"- {s} {p} {o}\n"
        context = facts_section

    prompt = f"""
You are a fact-checking assistant.
Your job is to verify a given claim using both the knowledge graph facts provided below and your own knowledge. 
If there are conflicting facts in the knowledge graph, prioritize logical reasoning and your internal knowledge to decide which fact is correct. 
Don't assume every KG fact is accurate — verify them.
{context}
Claim: "{claim}"
First, verify the relevant facts and resolve contradictions.
Then, answer in one word: SUPPORTED, REFUTED, or NOT ENOUGH INFO.
Explanation: Explain step-by-step why the claim is supported, refuted, or if there's not enough information.
Final Verdict:
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    max_tokens = 8192 - input_length

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.strip()

def chat_fn(message, history):
    output = check_claim_with_kg(message)
    history.append((message, output))
    return history, history

# === Custom Theme with Light Gray BG and Light Blue Buttons ===
custom_theme = gr.themes.Base(
    primary_hue="teal",
    secondary_hue="teal"
).set(
    body_background_fill="#f0f0f0",                      
    button_primary_background_fill="#40E0D0",            
    button_primary_background_fill_hover="#30cfc1",      
    button_primary_text_color="#000000",                 
    input_background_fill="#ffffff"                      
)

# === Interface Layout ===
with gr.Blocks(theme=custom_theme) as interface:
    gr.Markdown("<h1 style='text-align:center;'>Fake Claim Hunter 📰🔍</h1>")
    gr.Markdown("<p style='text-align:center;'>Enter a factual claim and click <b>Verify</b> to check if it's <i>SUPPORTED</i>, <i>REFUTED</i>, or there's <i>NOT ENOUGH INFO</i>.</p>")
    
    chatbot = gr.Chatbot(height=400)

    # Inject CSS to match heights of Textbox and Button
    gr.HTML("""
    <style>
        .input-row {
            display: flex;
            align-items: stretch;
            gap: 0.5rem;
        }
        .input-textbox {
            flex: 1;
        }
        .input-button {
            display: flex;
            align-items: center;
        }
        .input-button button {
            height: 100% !important;
        }
    </style>
    """)

    with gr.Row(elem_classes="input-row"):
        msg = gr.Textbox(
            show_label=False,
            placeholder="Type a factual claim like 'Water boils at 100 degree celcius'",
            container=False,
            elem_classes="input-textbox",
            scale=8
        )
        submit_btn = gr.Button("Verify", variant="primary", elem_classes="input-button",scale=1)

    state = gr.State([])  # keeps the conversation history
    submit_btn.click(chat_fn, [msg, state], [chatbot, state])
    msg.submit(chat_fn, [msg, state], [chatbot, state])
    with gr.Row():
        ex1 = gr.Button("Example 1: David Harbour is born on 2nd June")
        ex2 = gr.Button("Example 2: Paris is the capital of France")

    ex1.click(lambda: ("David Harbour is born on 2nd June", []), None, [msg, chatbot], queue=False)
    ex2.click(lambda: ("Paris is the capital of France", []), None, [msg, chatbot], queue=False)

# Launch app with public link
interface.launch(share=True)
