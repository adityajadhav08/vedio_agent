
# Actionable items, decisions, questions

# CHANGED: Replaced Mistral import with Gemini import
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
import re

def get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2
    )


def analyze_full_meeting(transcript: str) -> dict:
    llm = get_llm()
    system_prompt = (
        "You are an expert meeting analyst. Analyze the provided meeting transcript thoroughly "
        "and return a comprehensive analysis with the following 5 sections. "
        "Format your output exactly using the headers below:\n\n"
        "### TITLE\n"
        "A short professional meeting title (max 8 words).\n\n"
        "### SUMMARY\n"
        "A concise, professional executive summary in bullet points.\n\n"
        "### ACTION ITEMS\n"
        "Numbered list of action items including Task, Owner, and Deadline. If none, write 'No action items found.'\n\n"
        "### KEY DECISIONS\n"
        "Numbered list of key decisions made. If none, write 'No key decisions found.'\n\n"
        "### OPEN QUESTIONS\n"
        "Numbered list of unresolved questions or topics needing follow-up. If none, write 'No open questions found.'"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"text": transcript})

    def get_section(name, next_names):
        if next_names:
            pattern = rf"###\s*{name}\s*\n(.*?)(?=(?:###\s*(?:{'|'.join(next_names)}))|\Z)"
        else:
            pattern = rf"###\s*{name}\s*\n(.*)"
        match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    title = get_section("TITLE", ["SUMMARY", "ACTION ITEMS", "KEY DECISIONS", "OPEN QUESTIONS"])
    summary = get_section("SUMMARY", ["ACTION ITEMS", "KEY DECISIONS", "OPEN QUESTIONS"])
    actions = get_section("ACTION ITEMS", ["KEY DECISIONS", "OPEN QUESTIONS"])
    decisions = get_section("KEY DECISIONS", ["OPEN QUESTIONS"])
    questions = get_section("OPEN QUESTIONS", [])

    return {
        "title": title or "Meeting Analysis",
        "summary": summary or raw,
        "action_items": actions or "No action items found.",
        "key_decisions": decisions or "No key decisions found.",
        "open_questions": questions or "No open questions found."
    }


def build_chain(system_prompt : str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )


def extract_action_items(transcript:str)->str:
    chain = build_chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)
