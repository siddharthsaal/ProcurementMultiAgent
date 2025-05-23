import streamlit as st
import openai
import os
import time
import random
import requests
import json
import textwrap


import nest_asyncio
nest_asyncio.apply()

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from camel.agents import ChatAgent
from camel.toolkits import SearchToolkit
# from camel.workforce import Workforce # We might not use Workforce directly for sequential calls
# from camel.tasks import Task        # We might not use Task directly for sequential calls

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO

# --- LangChain Imports for Q&A ---
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from configparser import ConfigParser
import CamelUtils

config = ConfigParser()
config.read('config.ini')

utils = CamelUtils.CamelUtils()

# --- API Key Setups & Global Prompts ---

# Define ALL your persona_... example_... criteria_... strings here
# Example (ensure these are fully defined in your actual code):


OPENAI_API_KEY = config.get('API_KEYS', 'OPENAI_API_KEY')
PERPLEXITY_API_KEY = config.get('API_KEYS', 'PERPLEXITY_API_KEY')
llm_type = config.get('MODEL_CONFIG', 'LLM_TYPE')
model_name = config.get('MODEL_CONFIG', 'MODEL_NAME')
perplexity_api_url = config.get('MODEL_CONFIG', 'PERPLEXITY_API_URL')

# Reading General Prompts
perplexity_minimal_prompt = config.get('PROMPTS_GENERAL', 'PERPLEXITY_MINIMAL_SYSTEM_PROMPT')
not_found_phrase = config.get('PROMPTS_GENERAL', 'NOT_FOUND_IN_CONTEXT_PHRASE')


# Reading multi-line string prompts (configparser handles un-indentation)
persona_search = config.get('PROMPTS_SEARCH_AGENT', 'persona_search_agent')
example_search = config.get('PROMPTS_SEARCH_AGENT', 'example_search_agent')

SEARCH_AGENT_SYSTEM_MESSAGE=config.get('PROMPTS_SEARCH_AGENT','SEARCH_AGENT_SYSTEM_MESSAGE')
ADVANCED_SEARCH_PROMPT_TEMPLATE=config.get('PROMPTS_SEARCH_AGENT','ADVANCED_SEARCH_PROMPT_TEMPLATE')

CLASSIFICATION_SYSTEM_MESSAGE=config.get('PROMPTS_CLASSIFICATION_AGENT','CLASSIFICATION_SYSTEM_MESSAGE')
# You might want to apply textwrap.dedent() again if leading whitespace from the
# ini file's structure is an issue, though configparser usually handles it well.
# persona_search = textwrap.dedent(config.get('PROMPTS_SEARCH_AGENT', 'persona_search_agent')).strip()

ADVANCED_RFP_GENERATION_MESSAGE=config.get('PROMPTS_RFP_AGENT','ADVANCED_RFP_GENERATION_MESSAGE')

APPROVAL_AGENT_SYSTEM_MESSAGE=config.get('PROMPTS_APPROVAL_AGENT','APPROVAL_AGENT_SYSTEM_MESSAGE')


# Reading criteria (stored as JSON strings)
search_criteria_str = config.get('CRITERIA_JSON', 'search_and_ranking_criteria_json')
classification_criteria_str = config.get('CRITERIA_JSON', 'classification_criteria_json')
rfp_gen_criteria_str = config.get('CRITERIA_JSON', 'criteria_rfp_gen_json')

# Parsing JSON strings back into Python dictionaries
search_criteria = json.loads(search_criteria_str)
classification_criteria_dict = json.loads(classification_criteria_str) # Renamed to avoid conflict
rfp_gen_criteria = json.loads(rfp_gen_criteria_str)

import os
os.environ['OPENAI_API_KEY']=OPENAI_API_KEY


if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_KEY_HERE":
    st.error("🚨 OPENAI_API_KEY not set!"); st.stop()
if not PERPLEXITY_API_KEY or PERPLEXITY_API_KEY == "YOUR_PPLX_KEY_HERE":
    st.error("🚨 PERPLEXITY_API_KEY not set!"); st.stop()

# --- Model Initialization ---
try:
    shared_llm_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict={"temperature": 0.2}, # Lower temp for more factual agents
    )
    rfp_llm_model = ModelFactory.create( # Potentially a stronger model for RFP
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_TURBO, # Or GPT_4O if same
        model_config_dict={"temperature": 0.4},
    )
except Exception as e:
    st.error(f"Failed to create LLM models: {e}"); st.stop()

# --- Agent Creation Functions ---


# Instantiate Agents
search_agent = utils.make_chat_agent("Strategic Sourcing Analyst", SEARCH_AGENT_SYSTEM_MESSAGE, shared_llm_model)
classification_agent = utils.make_chat_agent("Product Classification Specialist", CLASSIFICATION_SYSTEM_MESSAGE, shared_llm_model)
rfp_agent = utils.make_chat_agent("RFP Generation Specialist", ADVANCED_RFP_GENERATION_MESSAGE, rfp_llm_model)
approval_agent = utils.make_chat_agent("Procurement Approval Officer", APPROVAL_AGENT_SYSTEM_MESSAGE, shared_llm_model) # New




# --- LangChain Q&A Setup (Re-introducing from previous correct version) ---

# --- Streamlit UI (Main Structure) ---
st.set_page_config(layout="wide", page_title="CAMEL AI Procurement Workflow")
st.title("🚀 CAMEL AI-Powered Procurement Pipeline")

qna_chain_combined_cached = utils.get_qna_chain_for_combined_context()

# Initialize session state (ensure this is at the top after imports and nest_asyncio.apply())
# ... (your session state initializations from previous example) ...
if "user_query_main" not in st.session_state: st.session_state.user_query_main = ""
if "user_query_details_for_agent" not in st.session_state: st.session_state.user_query_details_for_agent = {}
if "search_results" not in st.session_state: st.session_state.search_results = None
if "raw_perplexity_output" not in st.session_state: st.session_state.raw_perplexity_output = None # For debugging PPLX
if "classification_output" not in st.session_state: st.session_state.classification_output = None
if "rfp_content_markdown" not in st.session_state: st.session_state.rfp_content_markdown = None
if "approval_output" not in st.session_state: st.session_state.approval_output = None
if "rfp_approved_for_download" not in st.session_state: st.session_state.rfp_approved_for_download = False
if "current_step" not in st.session_state: st.session_state.current_step = 0
if "chat_history" not in st.session_state: st.session_state.chat_history = []


# --- Step 1: User Request Input ---
st.header("Step 1: User Procurement Request")
user_query_simple_input = st.text_area("Brief procurement need:", value=st.session_state.get("user_query_main", ""), height=70, key="uq_simple")
with st.expander("Optional: Add More Details for Advanced Search Prompt"):
    item_name_detail = st.text_input("Specific Item Name:", value=st.session_state.user_query_details_for_agent.get("item_name", ""), key="uq_item")
    mandatory_specs_detail = st.text_area("Key Mandatory Specs:", value=st.session_state.user_query_details_for_agent.get("mandatory_specs", ""), height=69, key="uq_mandspec")
    business_need_detail = st.text_area("Specific Business Need:", value=st.session_state.user_query_details_for_agent.get("business_need", ""), height=69, key="uq_bizneed")

if st.button("🚀 Submit & Start Search Agent", key="b_start_search", disabled=(st.session_state.current_step > 0)):
    if user_query_simple_input:
        st.session_state.user_query_main = user_query_simple_input
        # Construct the detailed dictionary for the ADVANCED_SEARCH_PROMPT_TEMPLATE
        details_for_search_prompt = {
            "request_id": f"REQ-{int(time.time())}", "department": "Demo User", "date": time.strftime("%Y-%m-%d"),
            "item_name": item_name_detail or user_query_simple_input.split(" for ")[0] if " for " in user_query_simple_input else user_query_simple_input,
            "business_need": business_need_detail or user_query_simple_input,
            "mandatory_specs": mandatory_specs_detail or "As described: " + user_query_simple_input,
            "desirable_specs": "High reliability, good support, cost-effective.", "quantity_timeline": "Initial assessment phase.",
            "item_name_for_sonar": item_name_detail or user_query_simple_input.split(" for ")[0] if " for " in user_query_simple_input else user_query_simple_input,
            "key_needs_for_sonar": mandatory_specs_detail or user_query_simple_input, "other_context_for_sonar": business_need_detail or ""
        }
        st.session_state.user_query_details_for_agent = details_for_search_prompt
        # Reset subsequent steps
        st.session_state.search_results = None; st.session_state.classification_output = None
        st.session_state.rfp_content_markdown = None; st.session_state.approval_output = None
        st.session_state.rfp_approved_for_download = False
        print("details for search prompt--")
        print(details_for_search_prompt)
        utils.trigger_search_agent_processing(details_for_search_prompt,shared_llm_model,rfp_llm_model)
        st.session_state.current_step = 1
        st.rerun()
    else:
        st.warning("Please enter procurement need.")
st.markdown("---")

# --- Step 2: Search Results & Classification Trigger ---
# --- Step 2: Search Results & Q&A & Classification Trigger ---
if st.session_state.current_step >= 1:
    st.header("Step 2: Search Output & Q&A")
    if st.session_state.get("raw_perplexity_output"):
        with st.expander("Raw Output (Debug)"):
            st.text_area("", value=st.session_state.raw_perplexity_output, height=150, disabled=True)
    
    if st.session_state.search_results:
        st.subheader("Search Agent Findings:")
        with st.container():
            st.markdown(f"""<div style="background-color:#f9f9f9;border:1px solid #e0e0e0;padding:10px;border-radius:5px;white-space:pre-wrap;max-height:250px;overflow-y:auto;">{st.session_state.search_results}</div>""", unsafe_allow_html=True)

        # --- Q&A Section ---
        st.subheader("💬 Ask Questions about Search Results or the RFP (once generated)")
        for chat_entry in st.session_state.chat_history:
            with st.chat_message(chat_entry["role"]):
                st.write(chat_entry["content"])
        
        user_chat_question = st.chat_input("Ask your question here...", key="chat_q_step2")

        if user_chat_question:
            st.session_state.chat_history.append({"role": "user", "content": user_chat_question})
            st.session_state.chat_history.append({"role": "assistant", "content": "Thinking..."})
            st.rerun()
        
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant" and st.session_state.chat_history[-1]["content"] == "Thinking...":
            actual_user_question = st.session_state.chat_history[-2]["content"]
            with st.spinner("Finding the best answer..."):
                answer = utils.answer_question_orchestrator(
                    qna_chain_combined_cached,
                    user_question=actual_user_question,
                    search_results=st.session_state.get("search_results"), # Pass current search results
                    rfp_markdown=st.session_state.get("rfp_content_markdown") ,# Pass RFP if available,
                    
                )
            st.session_state.chat_history[-1] = {"role": "assistant", "content": answer}
            st.rerun()
        # --- End Q&A Section ---

        if st.button("🔬 Proceed to Classify", key="b_classify_chat", disabled=(st.session_state.current_step > 1)):
            utils.trigger_classification_agent_processing(st.session_state.user_query_main, st.session_state.search_results,shared_llm_model,rfp_llm_model)
            st.session_state.current_step = 2 # Next step is classification display
            st.rerun()
    else:
        st.info("Search Agent is processing or encountered an issue. Please wait or check logs if it persists.")
st.markdown("---")


# --- Step 3: Classification & RFP Generation Trigger ---
if st.session_state.current_step >= 2:
    st.header("Step 3: Classification Agent Output")
    if st.session_state.classification_output:
        with st.container():
            st.markdown(f"""<div style="background-color:#f9f9f9;border:1px solid #e0e0e0;padding:10px;border-radius:5px;white-space:pre-wrap;">{st.session_state.classification_output}</div>""", unsafe_allow_html=True)
        if st.button("📝 Generate RFP Document", key="b_rfp_gen", disabled=(st.session_state.current_step > 2)):
            utils.trigger_rfp_agent_processing(
                st.session_state.user_query_main,
                st.session_state.classification_output,
                st.session_state.search_results,
                shared_llm_model,
                rfp_llm_model
            )
            st.session_state.current_step = 3
            st.rerun()
    else:
        st.info("Classification Agent is processing or encountered an issue.")
st.markdown("---")

# --- Step 4: Display RFP & Approval Trigger ---
if st.session_state.current_step >= 3:
    st.header("Step 4: RFP Generation Agent Output")
    if st.session_state.rfp_content_markdown:
        st.subheader("Draft Request for Proposal (RFP):")
        with st.container():
            st.markdown(f"""<div style="background-color:#fdfaed;border:1px solid #f0e68c;padding:15px;border-radius:5px;white-space:pre-wrap;max-height:400px;overflow-y:auto;">{st.session_state.rfp_content_markdown}</div>""", unsafe_allow_html=True)

        if st.button("⚖️ Submit RFP for Approval", key="b_submit_approval", disabled=(st.session_state.current_step > 3)):
            utils.trigger_approval_agent_processing(
                st.session_state.user_query_main,
                st.session_state.search_results,
                st.session_state.classification_output,
                st.session_state.rfp_content_markdown,
                shared_llm_model,
                rfp_llm_model
            )
            st.session_state.current_step = 4
            st.rerun()
    else:
        st.info("RFP Generation Agent is processing or encountered an issue.")
st.markdown("---")

# --- Step 5: Display Approval & PDF Download ---
if st.session_state.current_step >= 4:
    st.header("Step 5: Approval Officer Decision")
    if st.session_state.approval_output:
        st.subheader("Approval Status & Justification:")
        with st.container():
            st.markdown(f"""<div style="background-color:#e6f3ff;border:1px solid #cce0ff;padding:15px;border-radius:5px;white-space:pre-wrap;">{st.session_state.approval_output}</div>""", unsafe_allow_html=True)

        is_approved = "Decision: Approve" in st.session_state.approval_output # Simple check
        st.session_state.rfp_approved_for_download = is_approved # Update flag

        if is_approved:
            st.success("🎉 RFP Approved by Approval Agent!")
            if st.session_state.rfp_content_markdown:
                pdf_buffer = utils.create_pdf_from_markdown(st.session_state.rfp_content_markdown)
                if pdf_buffer:
                    item_name_for_file = st.session_state.user_query_details_for_agent.get('item_name', 'Procurement').replace(' ', '_').replace('/', '_')
                    st.download_button(
                        label="📥 Download Approved RFP as PDF", data=pdf_buffer,
                        file_name=f"Approved_RFP_{item_name_for_file}.pdf", mime="application/pdf",
                        key="download_approved_rfp_pdf"
                    )
            else:
                st.warning("RFP content missing, cannot generate PDF for download.")
        else:
            st.warning("🚦 RFP requires revision based on Approval Agent's feedback.")
    else:
        st.info("Approval Officer is processing or encountered an issue.")
st.markdown("---")

# --- Reset Button ---
if st.button("🔄 Reset Entire Workflow", key="reset_all"):
    # Clear all relevant session state keys
    keys_to_clear = [
        "user_query_main", "user_query_details_for_agent", "search_results",
        "raw_perplexity_output", "classification_output", "rfp_content_markdown",
        "approval_output", "rfp_approved_for_download", "current_step"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()